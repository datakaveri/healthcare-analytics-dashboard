import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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

def find_optimal_clusters(data, max_k=8):
    """Find optimal number of clusters using silhouette score"""
    silhouette_scores = []
    K_range = range(2, max_k+1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    return K_range[np.argmax(silhouette_scores)]

def create_comprehensive_radar_chart(cluster_centers, optimal_k, df, top_n=8):
    """Create radar chart showing top symptoms for each cluster"""
    
    # Calculate subplot layout
    cols = 2
    rows = (optimal_k + 1) // 2
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Cluster {i} Profile ({len(df[df.cluster==i])} patients)' 
                       for i in range(optimal_k)],
        specs=[[{'type': 'polar'} for _ in range(cols)] for _ in range(rows)]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for i in range(optimal_k):
        cluster_data = cluster_centers.iloc[i]
        
        # Get top symptoms (highest absolute values)
        top_symptoms = cluster_data.abs().nlargest(top_n)
        
        # Clean symptom names for better readability
        theta = []
        for col in top_symptoms.index:
            if '|' in col:
                clean_name = col.split('|')[-1].strip()
            else:
                clean_name = col
            # Truncate long names
            if len(clean_name) > 20:
                clean_name = clean_name[:17] + "..."
            theta.append(clean_name)
        
        r = [cluster_data[col] for col in top_symptoms.index]
        
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill='toself',
            name=f'Cluster {i}',
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.6,
            line=dict(width=2)
        ), row=row, col=col)
    
    fig.update_layout(
        title=dict(
            text=f"Patient Phenotypes: Symptom Profiles Across {optimal_k} Clusters",
            x=0.5,
            font=dict(size=16, color='darkblue')
        ),
        height=400 * rows,
        width=800,
        showlegend=False,
        font=dict(size=10)
    )
    
    return fig

def phenotyping_analysis_page(processed_data):
    """Main patient phenotyping analysis page"""
    
    st.markdown('<h2 class="section-header">🧬 Patient Phenotyping Analysis</h2>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
    <strong>🧬 Patient Phenotyping:</strong> This analysis identifies distinct patient phenotypes based on symptom patterns. 
    Each phenotype represents a unique combination of symptoms that can help in personalized treatment approaches.
    </div>
    """, unsafe_allow_html=True)
    
    # Identify symptom columns (exclude patient info columns)
    exclude_cols = ['patient_id', 'gender', 'active', 'last_updated']
    symptom_cols = [col for col in processed_data.columns if col not in exclude_cols]
    
    # Prepare and standardize symptom data
    symptom_data_raw = processed_data[symptom_cols].copy()
    symptom_data, dropped_cols = _coerce_numeric_frame(symptom_data_raw)
    if dropped_cols:
        st.warning(
            f"Dropped {len(dropped_cols)} non-numeric column(s) from phenotyping "
            f"(e.g. `{dropped_cols[0]}`) to prevent numeric conversion errors."
        )
    symptom_data = symptom_data.fillna(0)
    if symptom_data.shape[1] < 2:
        st.error(
            "Not enough numeric columns available for phenotyping after cleaning. "
            "Please check your dataset columns."
        )
        return
    scaler = StandardScaler()
    symptom_scaled = scaler.fit_transform(symptom_data)
    
    # Find optimal number of clusters using silhouette score
    # Guard against very small datasets
    max_k = min(8, max(2, len(symptom_scaled) - 1))
    optimal_k = find_optimal_clusters(symptom_scaled, max_k=max_k)
    
    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(symptom_scaled)
    
    # Add cluster labels to dataframe
    df = processed_data.copy()
    df['cluster'] = cluster_labels
    
    # Create cluster centers dataframe
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_, 
        columns=list(symptom_data.columns),
        index=[f'Cluster {i}' for i in range(optimal_k)]
    )
    
    # Create the radar chart
    radar_fig = create_comprehensive_radar_chart(cluster_centers, optimal_k, df)
    st.plotly_chart(radar_fig, use_container_width=True)
    
    # Phenotyping summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Optimal Clusters", optimal_k)
    
    with col2:
        st.metric("Total Patients", len(df))
    
    with col3:
        st.metric("Largest Phenotype", f"{df['cluster'].value_counts().max()} patients")
    
    with col4:
        st.metric("Smallest Phenotype", f"{df['cluster'].value_counts().min()} patients")
    
    # Cluster details
    st.markdown('<h3 class="section-header">Phenotype Details</h3>', unsafe_allow_html=True)
    
    # Cluster sizes and gender distribution
    cluster_sizes = df['cluster'].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Phenotype Sizes:**")
        cluster_sizes_df = pd.DataFrame({
            'Phenotype': cluster_sizes.index,
            'Size': cluster_sizes.values,
            'Percentage': (cluster_sizes.values / len(df) * 100).round(1)
        })
        st.dataframe(cluster_sizes_df, use_container_width=True)
    
    with col2:
        st.markdown("**Gender Distribution by Phenotype:**")
        gender_dist = df.groupby(['cluster', 'gender']).size().unstack(fill_value=0)
        gender_dist.index = [f'Phenotype {i}' for i in gender_dist.index]
        st.dataframe(gender_dist, use_container_width=True)
    
    # Detailed phenotype analysis
    st.markdown('<h3 class="section-header">Detailed Phenotype Analysis</h3>', unsafe_allow_html=True)
    
    # Create tabs for each phenotype
    tabs = st.tabs([f"Phenotype {i}" for i in range(optimal_k)])
    
    for i, tab in enumerate(tabs):
        with tab:
            cluster_patients = df[df['cluster'] == i]
            cluster_profile = cluster_centers.iloc[i]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Phenotype {i} Overview:**")
                st.write(f"• **Size:** {len(cluster_patients)} patients")
                st.write(f"• **Gender:** {dict(cluster_patients['gender'].value_counts())}")
                
                # Top symptoms
                top_symptoms = cluster_profile.nlargest(5)
                st.write("• **Key symptoms:**")
                for symptom, score in top_symptoms.items():
                    clean_name = symptom.split('|')[-1].strip() if '|' in symptom else symptom
                    st.write(f"  - {clean_name}: {score:.2f}")
            
            with col2:
                st.markdown("**Patient List:**")
                patient_list = cluster_patients[['patient_id', 'gender']].copy()
                patient_list['patient_id'] = patient_list['patient_id'].str[:8] + "..."
                st.dataframe(patient_list, use_container_width=True)
    
    # Phenotype comparison heatmap
    st.markdown('<h3 class="section-header">Phenotype Comparison Heatmap</h3>', unsafe_allow_html=True)
    
    # Get top symptoms across all phenotypes
    all_symptoms = []
    for i in range(optimal_k):
        top_symptoms = cluster_centers.iloc[i].nlargest(10)
        all_symptoms.extend(top_symptoms.index)
    
    # Get unique symptoms
    unique_symptoms = list(set(all_symptoms))[:15]  # Top 15 for readability
    
    # Create heatmap data
    heatmap_data = cluster_centers[unique_symptoms].T
    
    # Clean symptom names for display
    clean_symptom_names = []
    for symptom in unique_symptoms:
        if '|' in symptom:
            clean_name = symptom.split('|')[-1].strip()
        else:
            clean_name = symptom
        if len(clean_name) > 30:
            clean_name = clean_name[:27] + "..."
        clean_symptom_names.append(clean_name)
    
    heatmap_data.index = clean_symptom_names
    
    # Create heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f'Phenotype {i}' for i in range(optimal_k)],
        y=heatmap_data.index,
        colorscale='Blues',
        text=np.round(heatmap_data.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title="Phenotype Symptom Profiles Comparison",
        xaxis_title="Phenotypes",
        yaxis_title="Symptoms",
        width=800,
        height=600,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
