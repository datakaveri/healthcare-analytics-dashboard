import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
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

def create_dendrogram_plot(data, labels=None, title="Dendrogram"):
    """Create an interactive dendrogram plot"""
    
    if len(data) < 2:
        return None
    
    # Calculate distance matrix
    try:
        # Use only numeric columns (and coerce numeric-like strings)
        numeric_data_raw = data.select_dtypes(include=[np.number, "bool"]).copy()
        # If caller passed mixed types, also try coercing everything numeric-like
        if numeric_data_raw.empty:
            numeric_data_raw = data.copy()
        numeric_data, _ = _coerce_numeric_frame(numeric_data_raw)
        
        if numeric_data.empty:
            return None
            
        # Calculate pairwise distances
        distances = pdist(numeric_data.values, metric='euclidean')
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method='ward')
        
        # Create labels if not provided
        if labels is None:
            labels = [f"Patient {i+1}" for i in range(len(data))]
        
        # Create dendrogram
        fig = go.Figure()
        
        # Add dendrogram traces
        dendro = dendrogram(linkage_matrix, labels=labels, no_plot=True)
        
        # Create dendrogram visualization
        fig.add_trace(go.Scatter(
            x=dendro['icoord'],
            y=dendro['dcoord'],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add labels
        fig.add_trace(go.Scatter(
            x=dendro['leaves'],
            y=[0] * len(dendro['leaves']),
            mode='markers+text',
            text=dendro['ivl'],
            textposition='top center',
            marker=dict(size=8, color='red'),
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Patients',
            yaxis_title='Distance',
            height=600,
            font=dict(size=12)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating dendrogram: {e}")
        return None

def create_advanced_heatmap(data, title="Heatmap", color_scale='Blues'):
    """Create an advanced interactive heatmap"""
    
    try:
        # Ensure numeric heatmap values (avoid crashing on strings)
        numeric_data, dropped_cols = _coerce_numeric_frame(data.copy())
        numeric_data = numeric_data.fillna(0)
        if numeric_data.empty:
            return None

        # Clean column names for better display
        clean_cols = []
        for col in numeric_data.columns:
            if '|' in col:
                clean_name = col.split('|')[-1].strip()
            else:
                clean_name = col
            if len(clean_name) > 20:
                clean_name = clean_name[:17] + "..."
            clean_cols.append(clean_name)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=numeric_data.values,
            x=clean_cols,
            y=numeric_data.index,
            colorscale=color_scale,
            text=np.round(numeric_data.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hoverlabel=dict(bgcolor="white", font_size=12)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Variables",
            yaxis_title="Patients",
            height=max(400, len(numeric_data) * 20),
            width=max(600, len(numeric_data.columns) * 30),
            font=dict(size=12)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")
        return None

def create_cluster_heatmap(data, cluster_labels, title="Cluster Analysis Heatmap"):
    """Create a heatmap with cluster information"""
    
    try:
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['Cluster'] = cluster_labels
        
        # Sort by cluster
        data_sorted = data_with_clusters.sort_values('Cluster')
        
        # Remove cluster column for heatmap
        heatmap_data = data_sorted.drop('Cluster', axis=1)
        
        # Create heatmap
        fig = create_advanced_heatmap(heatmap_data, title, 'RdYlBu_r')
        
        # Add cluster annotations
        cluster_boundaries = []
        current_cluster = None
        for i, cluster in enumerate(data_sorted['Cluster']):
            if current_cluster != cluster:
                if current_cluster is not None:
                    cluster_boundaries.append(i)
                current_cluster = cluster
        
        # Add vertical lines for cluster boundaries
        for boundary in cluster_boundaries:
            fig.add_vline(
                x=boundary,
                line_dash="dash",
                line_color="red",
                opacity=0.7
            )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating cluster heatmap: {e}")
        return None

def create_symptom_network(data, threshold=0.3):
    """Create a symptom correlation network"""
    
    try:
        # Calculate correlation matrix on numeric-only data
        numeric_data, _ = _coerce_numeric_frame(data.copy())
        if numeric_data.shape[1] < 2:
            return None
        corr_matrix = numeric_data.corr()
        
        # Get strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    strong_corr.append({
                        'source': corr_matrix.columns[i],
                        'target': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if not strong_corr:
            return None
        
        # Create network plot
        fig = go.Figure()
        
        # Add edges
        for edge in strong_corr:
            fig.add_trace(go.Scatter(
                x=[edge['source'], edge['target']],
                y=[0, 0],
                mode='lines',
                line=dict(
                    width=abs(edge['correlation']) * 5,
                    color='red' if edge['correlation'] < 0 else 'blue'
                ),
                opacity=0.6,
                showlegend=False,
                hoverinfo='text',
                hovertext=f"Correlation: {edge['correlation']:.3f}"
            ))
        
        fig.update_layout(
            title=f"Symptom Correlation Network (threshold: {threshold})",
            xaxis_title="Symptoms",
            yaxis_title="",
            height=400,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating network plot: {e}")
        return None

def create_time_series_plot(data, time_col=None, value_cols=None):
    """Create time series plots for temporal data"""
    
    try:
        if time_col is None or value_cols is None:
            return None
        
        fig = go.Figure()
        
        for col in value_cols:
            fig.add_trace(go.Scatter(
                x=data[time_col],
                y=data[col],
                mode='lines+markers',
                name=col,
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Time Series Analysis",
            xaxis_title=time_col,
            yaxis_title="Values",
            height=500,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating time series plot: {e}")
        return None
