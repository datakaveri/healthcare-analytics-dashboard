import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
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

def clustering_analysis_page(processed_data):
    """Main clustering analysis page"""
    
    st.markdown('<h2 class="section-header">🎯 Patient Clustering Analysis</h2>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
    <strong>🎯 Clustering Analysis:</strong> This analysis groups patients based on similar symptom patterns using K-means clustering. 
    PCA (Principal Component Analysis) is used to visualize the clusters in 2D space.
    </div>
    """, unsafe_allow_html=True)
    
    # Get symptom columns (exclude patient info columns)
    exclude_cols = ['patient_id', 'gender', 'active', 'last_updated']
    symptom_cols = [col for col in processed_data.columns if col not in exclude_cols]
    
    # Prepare data
    X_raw = processed_data[symptom_cols].copy()
    X, dropped_cols = _coerce_numeric_frame(X_raw)
    if dropped_cols:
        st.warning(
            f"Dropped {len(dropped_cols)} non-numeric column(s) from clustering "
            f"(e.g. `{dropped_cols[0]}`) to prevent numeric conversion errors."
        )
    X = X.fillna(0)
    if X.shape[1] < 2:
        st.error(
            "Not enough numeric columns available for clustering after cleaning. "
            "Please check your dataset columns."
        )
        return
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create results dataframe
    results_df = processed_data.copy()
    results_df['Cluster'] = clusters
    results_df['PCA1'] = X_pca[:, 0]
    results_df['PCA2'] = X_pca[:, 1]
    
    # Calculate cluster statistics
    # Only compute stats on numeric features actually used
    cluster_stats = results_df.groupby('Cluster')[list(X.columns)].mean(numeric_only=True)
    cluster_sizes = results_df['Cluster'].value_counts().sort_index()
    
    # Create interactive scatter plot
    fig = px.scatter(
        results_df, 
        x='PCA1', 
        y='PCA2', 
        color='Cluster',
        symbol='gender',
        title='Patient Clustering Based on Symptom Patterns',
        hover_data=['patient_id', 'gender'],
        color_discrete_sequence=px.colors.qualitative.Set2,
        width=900,
        height=600
    )
    
    # Add cluster centers
    centers_pca = pca.transform(kmeans.cluster_centers_)
    fig.add_trace(go.Scatter(
        x=centers_pca[:, 0],
        y=centers_pca[:, 1],
        mode='markers',
        marker=dict(symbol='x', size=15, color='black', line=dict(width=2)),
        name='Cluster Centers',
        hovertemplate='Cluster Center %{pointNumber}<extra></extra>'
    ))
    
    fig.update_layout(
        title_font_size=16,
        xaxis_title=f'First Principal Component (explains {pca.explained_variance_ratio_[0]:.1%} variance)',
        yaxis_title=f'Second Principal Component (explains {pca.explained_variance_ratio_[1]:.1%} variance)',
        legend_title='Patient Clusters',
        template='plotly_white',
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Variance Explained", f"{sum(pca.explained_variance_ratio_):.1%}")
    
    with col2:
        st.metric("Number of Clusters", len(cluster_sizes))
    
    with col3:
        st.metric("Largest Cluster", f"{cluster_sizes.max()} patients")
    
    with col4:
        st.metric("Smallest Cluster", f"{cluster_sizes.min()} patients")
    
    # Cluster details
    st.markdown('<h3 class="section-header">Cluster Details</h3>', unsafe_allow_html=True)
    
    # Cluster sizes table
    cluster_sizes_df = pd.DataFrame({
        'Cluster': cluster_sizes.index,
        'Size': cluster_sizes.values,
        'Percentage': (cluster_sizes.values / len(results_df) * 100).round(1)
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cluster Sizes:**")
        st.dataframe(cluster_sizes_df, use_container_width=True)
    
    with col2:
        st.markdown("**Gender Distribution by Cluster:**")
        gender_dist = results_df.groupby(['Cluster', 'gender']).size().unstack(fill_value=0)
        st.dataframe(gender_dist, use_container_width=True)
    
    # Top symptoms by cluster
    st.markdown('<h3 class="section-header">Top Symptoms by Cluster</h3>', unsafe_allow_html=True)
    
    cluster_symptoms = []
    for cluster in range(len(cluster_sizes)):
        top_symptoms = cluster_stats.loc[cluster].nlargest(5)
        for symptom, score in top_symptoms.items():
            # Clean symptom name
            clean_name = symptom.split('|')[-1].strip() if '|' in symptom else symptom
            cluster_symptoms.append({
                'Cluster': cluster,
                'Symptom': clean_name,
                'Average Score': round(score, 3)
            })
    
    cluster_symptoms_df = pd.DataFrame(cluster_symptoms)
    
    # Create tabs for each cluster
    tabs = st.tabs([f"Cluster {i}" for i in range(len(cluster_sizes))])
    
    for i, tab in enumerate(tabs):
        with tab:
            cluster_data = cluster_symptoms_df[cluster_symptoms_df['Cluster'] == i]
            st.dataframe(cluster_data[['Symptom', 'Average Score']], use_container_width=True)
    
