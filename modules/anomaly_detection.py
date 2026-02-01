import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st

def _coerce_numeric_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Coerce all columns to numeric where possible.
    Non-numeric values become NaN; columns that are entirely NaN are dropped.
    Returns (numeric_df, dropped_columns).
    """
    if df.empty:
        return df.copy(), []

    numeric = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    dropped = numeric.columns[numeric.isna().all()].tolist()
    numeric = numeric.drop(columns=dropped)
    return numeric, dropped

def anomaly_detection_page(processed_data):
    """Main anomaly detection page"""
    
    st.markdown('<h2 class="section-header">🚨 Anomaly Detection Analysis</h2>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
    <strong>🚨 Anomaly Detection:</strong> This analysis identifies unusual patient cases using Isolation Forest algorithm. 
    Anomalous patients may have rare symptom combinations or unusual patterns that warrant further investigation.
    </div>
    """, unsafe_allow_html=True)
    
    # Select symptom/feature columns (avoid forcing non-numeric data into the model)
    exclude_cols = ['patient_id', 'gender', 'active', 'last_updated']
    candidate_cols = [c for c in processed_data.columns if c not in exclude_cols]
    symptom_cols = candidate_cols
    X_raw = processed_data[symptom_cols].copy()
    X, dropped_cols = _coerce_numeric_frame(X_raw)
    if dropped_cols:
        st.warning(
            f"Dropped {len(dropped_cols)} non-numeric feature column(s) for anomaly detection "
            f"(e.g. `{dropped_cols[0]}`) to prevent numeric conversion errors."
        )
    X = X.fillna(0)
    
    # Add some derived features for better anomaly detection
    df = processed_data.copy()
    df['total_symptoms'] = pd.to_numeric(X.sum(axis=1), errors="coerce").fillna(0.0)
    df['symptom_diversity'] = pd.to_numeric((X > 0).sum(axis=1), errors="coerce").fillna(0.0)  # Number of different symptoms
    df['severity_ratio'] = (df['total_symptoms'] / (df['symptom_diversity'] + 1)).astype(float)  # Avoid division by zero
    
    # Prepare features for anomaly detection
    feature_cols = list(X.columns) + ['total_symptoms', 'symptom_diversity', 'severity_ratio']
    X_features_raw = df[feature_cols].copy()
    X_features, dropped_feature_cols = _coerce_numeric_frame(X_features_raw)
    X_features = X_features.fillna(0)
    if X_features.shape[1] < 2:
        st.error(
            "Not enough numeric features available for anomaly detection after cleaning. "
            "Please check your dataset columns."
        )
        return
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.decision_function(X_scaled)
    
    # Add results to dataframe
    df['anomaly'] = anomaly_labels
    df['anomaly_score'] = anomaly_scores
    df['is_anomaly'] = df['anomaly'] == -1
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    # Create main visualization
    fig = px.scatter(
        df,
        x='PCA1',
        y='PCA2',
        color='is_anomaly',
        size='total_symptoms',
        symbol='gender',
        hover_data=['patient_id', 'total_symptoms', 'symptom_diversity'],
        title='Anomaly Detection: Identifying Unusual Patient Cases',
        color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'},
        labels={'is_anomaly': 'Anomalous Case'},
        width=900,
        height=600
    )
    
    # Customize hover template
    fig.update_traces(
        hovertemplate='<b>Patient:</b> %{customdata[0]}<br>' +
                      '<b>Gender:</b> %{marker.symbol}<br>' +
                      '<b>Total Symptoms:</b> %{customdata[1]}<br>' +
                      '<b>Symptom Diversity:</b> %{customdata[2]}<br>' +
                      '<b>Anomaly:</b> %{color}<br>' +
                      '<extra></extra>'
    )
    
    fig.update_layout(
        title_font_size=16,
        xaxis_title=f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)',
        yaxis_title=f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)',
        template='plotly_white',
        legend_title='Patient Type',
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly summary metrics
    anomalous_patients = df[df['is_anomaly'] == True]
    normal_patients = df[df['is_anomaly'] == False]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", len(df))
    
    with col2:
        st.metric("Anomalous Cases", len(anomalous_patients))
    
    with col3:
        st.metric("Anomaly Rate", f"{len(anomalous_patients)/len(df):.1%}")
    
    with col4:
        st.metric("Normal Cases", len(normal_patients))
    
    # Analyze the anomalous cases
    if len(anomalous_patients) > 0:
        # Find most common symptoms in anomalous vs normal patients
        symptom_feature_cols = list(X.columns)
        anomaly_symptom_rates = anomalous_patients[symptom_feature_cols].mean(numeric_only=True)
        normal_symptom_rates = normal_patients[symptom_feature_cols].mean(numeric_only=True)
        symptom_difference = anomaly_symptom_rates - normal_symptom_rates
        
        # Get top differentiating symptoms
        top_differentiating = symptom_difference.abs().nlargest(10)
        
        # Create comparison plot
        fig_comparison = go.Figure()
        
        symptoms_to_plot = top_differentiating.index[:8]  # Top 8 for readability
        anomaly_rates = [anomaly_symptom_rates[s] for s in symptoms_to_plot]
        normal_rates = [normal_symptom_rates[s] for s in symptoms_to_plot]
        
        # Clean symptom names
        clean_symptom_names = []
        for symptom in symptoms_to_plot:
            if '|' in symptom:
                clean_name = symptom.split('|')[-1].strip()
            else:
                clean_name = symptom
            clean_symptom_names.append(clean_name[:20])  # Limit length
        
        fig_comparison.add_trace(go.Bar(
            name='Normal Patients',
            x=clean_symptom_names,
            y=normal_rates,
            marker_color='#4ECDC4',
            opacity=0.8
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Anomalous Patients',
            x=clean_symptom_names,
            y=anomaly_rates,
            marker_color='#FF6B6B',
            opacity=0.8
        ))
        
        fig_comparison.update_layout(
            title='Symptom Rates: Normal vs Anomalous Patients',
            xaxis_title='Symptoms',
            yaxis_title='Prevalence Rate',
            barmode='group',
            template='plotly_white',
            width=900,
            height=500,
            xaxis_tickangle=-45,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Anomalous patients details
        st.markdown('<h3 class="section-header">Anomalous Patients Details</h3>', unsafe_allow_html=True)
        
        # Most anomalous patients
        most_anomalous = anomalous_patients.nsmallest(3, 'anomaly_score')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Most Anomalous Patients:**")
            for _, patient in most_anomalous.iterrows():
                st.write(f"• Patient {patient['patient_id'][:8]}... ({patient['gender']}): {patient['total_symptoms']} symptoms, Score: {patient['anomaly_score']:.3f}")
        
        with col2:
            st.markdown("**Characteristics Comparison:**")
            st.write(f"**Average total symptoms:**")
            st.write(f"  • Anomalous: {anomalous_patients['total_symptoms'].mean():.1f}")
            st.write(f"  • Normal: {normal_patients['total_symptoms'].mean():.1f}")
            
            st.write(f"**Average symptom diversity:**")
            st.write(f"  • Anomalous: {anomalous_patients['symptom_diversity'].mean():.1f}")
            st.write(f"  • Normal: {normal_patients['symptom_diversity'].mean():.1f}")
        
        # Symptoms more common in anomalous patients
        st.markdown('<h3 class="section-header">Symptoms More Common in Anomalous Patients</h3>', unsafe_allow_html=True)
        
        increased_symptoms = symptom_difference[symptom_difference > 0.1].nlargest(5)
        if len(increased_symptoms) > 0:
            increased_df = pd.DataFrame({
                'Symptom': [s.split('|')[-1] if '|' in s else s for s in increased_symptoms.index],
                'Difference': [f"+{diff:.1%}" for diff in increased_symptoms.values]
            })
            st.dataframe(increased_df, use_container_width=True)
        else:
            st.info("No symptoms show significant increase in anomalous patients.")
        
        # Anomalous patients table
        st.markdown('<h3 class="section-header">Anomalous Patients Table</h3>', unsafe_allow_html=True)
        
        anomalous_display = anomalous_patients[['patient_id', 'gender', 'total_symptoms', 'symptom_diversity', 'anomaly_score']].copy()
        anomalous_display['anomaly_score'] = anomalous_display['anomaly_score'].round(4)
        
        st.dataframe(anomalous_display, use_container_width=True)
    
    else:
        st.info("No anomalous patients detected in this dataset.")
