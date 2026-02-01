import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def _coerce_numeric_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Coerce columns to numeric where possible.
    Non-numeric values become NaN; columns entirely NaN are dropped.
    """
    if df.empty:
        return df.copy(), []
    numeric = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    dropped = numeric.columns[numeric.isna().all()].tolist()
    numeric = numeric.drop(columns=dropped)
    return numeric, dropped

def _presence_or_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a series into a numeric vector suitable for correlation.
    - If values are numeric-like, return numeric values.
    - If values are non-numeric (e.g., strings like 'Labour'), return presence indicator (1 if present).
    """
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().sum() > 0:
        return num
    # Presence indicator for categorical/string observations
    s = series
    present = s.notna() & (s.astype(str).str.strip() != "")
    return present.astype(int)

def _build_obs_column_map(df: pd.DataFrame, obs_names: list[str]) -> dict[str, list[str]]:
    """
    Map each observation name to its exact/numbered column(s) in the DataFrame.
    
    obs_names contains the FULL piped names from FHIR (e.g., "House (environment) | Count of entities...")
    We need to find columns that:
    1. Match exactly
    2. Match with numbered suffix like " (2)", " (3)"
    """
    colset = set(df.columns)
    obs_map: dict[str, list[str]] = {}
    
    for name in obs_names or []:
        cols = []
        
        # Exact match
        if name in colset:
            cols.append(name)
        
        # Check for numbered variants: "name (2)", "name (3)", etc.
        i = 2
        while True:
            numbered_name = f"{name} ({i})"
            if numbered_name in colset:
                cols.append(numbered_name)
                i += 1
            else:
                break
        
        if cols:
            obs_map[name] = sorted(cols)
    
    return obs_map

def _build_cond_column_map(df: pd.DataFrame, cond_names: list[str]) -> dict[str, str]:
    """
    Map each condition name to its exact column.
    Conditions are binary (0/1) and don't have numbered variants.
    """
    colset = set(df.columns)
    return {name: name for name in (cond_names or []) if name in colset}


def plot_correlation(processed_data, obs_names, cond_names, obs_column_map, cond_column_map):
    """Create correlation analysis with interactive heatmaps"""
    # Build the feature set from explicit FHIR origin lists (Observations + Conditions)
    exclude_cols = {"patient_id", "gender", "active", "last_updated"}

    # Get all observation columns (including numbered variants)
    observation_columns = sorted({c for cols in (obs_column_map or {}).values() for c in cols if c not in exclude_cols})
    # Get all condition columns
    condition_columns = sorted({c for c in (cond_column_map or {}).values() if c not in exclude_cols})

    # Fallback: if no explicit mapping found, use all non-id columns
    if not observation_columns and not condition_columns:
        candidate_cols = [c for c in processed_data.columns if c not in exclude_cols]
    else:
        candidate_cols = observation_columns + condition_columns

    if len(candidate_cols) < 2:
        return None, None

    # Convert to numeric/presence for correlation
    df_corr = pd.DataFrame({c: _presence_or_numeric(processed_data[c]) for c in candidate_cols})
    df_corr = df_corr.fillna(0)

    # Drop constant columns (corr undefined)
    constant_value_cols = [c for c in df_corr.columns if df_corr[c].nunique(dropna=False) <= 1]
    if constant_value_cols:
        df_corr = df_corr.drop(columns=constant_value_cols)

    if df_corr.shape[1] < 2:
        return None, None

    # If we have both observations and conditions, create cross-correlation matrix
    if len(observation_columns) > 0 and len(condition_columns) > 0:
        correlation_matrix = df_corr.corr()
        
        # Only include those that survived constant-column dropping
        obs_in = [c for c in observation_columns if c in correlation_matrix.index]
        cond_in = [c for c in condition_columns if c in correlation_matrix.columns]
        
        if not obs_in or not cond_in:
            # Fall back to full correlation matrix
            fig = px.imshow(correlation_matrix,
                            labels=dict(x="Variables", y="Variables", color="Correlation"),
                            title="Correlation Matrix: All Variables",
                            color_continuous_scale='RdBu',
                            aspect="auto",
                            text_auto='.2f')
            
            fig.update_layout(
                width=max(800, len(correlation_matrix.columns) * 30),
                height=max(600, len(correlation_matrix.index) * 30),
                title_x=0.5,
                font=dict(size=10),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_coloraxes(cmin=-1, cmax=1, cmid=0)
            
            return fig, correlation_matrix
        
        cross_corr = correlation_matrix.loc[obs_in, cond_in]
        
        # Create clean labels for display (take last part after |)
        clean_obs_names = []
        for obs_col in obs_in:
            if '|' in obs_col:
                parts = obs_col.split('|')
                clean_name = parts[-1].strip()
            else:
                clean_name = obs_col
            # Preserve numbered suffix
            clean_obs_names.append(clean_name)
        
        clean_cond_names = []
        for cond_col in cond_in:
            if '|' in cond_col:
                parts = cond_col.split('|')
                clean_name = parts[-1].strip()
            else:
                clean_name = cond_col
            clean_cond_names.append(clean_name)
        
        cross_corr.index = clean_obs_names
        cross_corr.columns = clean_cond_names
        
        fig = px.imshow(cross_corr,
                        labels=dict(x="Conditions", y="Observations", color="Correlation"),
                        title="Correlation Matrix: Observations vs Conditions",
                        color_continuous_scale='RdBu',
                        aspect="auto",
                        text_auto='.2f')
        
        fig.update_layout(
            width=max(800, len(cond_in) * 50),
            height=max(600, len(obs_in) * 30),
            title_x=0.5,
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_coloraxes(cmin=-1, cmax=1, cmid=0)
        
        return fig, cross_corr
        
    else:
        # Single type or mixed - show full correlation matrix
        correlation_matrix = df_corr.corr()
        
        fig = px.imshow(correlation_matrix,
                        labels=dict(x="Variables", y="Variables", color="Correlation"),
                        title="Correlation Matrix: All Variables",
                        color_continuous_scale='RdBu',
                        aspect="auto",
                        text_auto='.2f')
        
        fig.update_layout(
            width=max(800, len(correlation_matrix.columns) * 30),
            height=max(600, len(correlation_matrix.index) * 30),
            title_x=0.5,
            font=dict(size=10),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_coloraxes(cmin=-1, cmax=1, cmid=0)
        
        return fig, correlation_matrix


def plot_observation_data(processed_data, obs_names, obs_column_map):
    """Plot observation data analysis (FHIR Observations)"""
    
    observation_data = {}

    # Get all observation columns from the map (includes numbered variants)
    obs_cols = sorted({c for cols in (obs_column_map or {}).values() for c in cols})
    
    for col in obs_cols:
        if col not in processed_data.columns:
            continue
        s = processed_data[col]
        # Count how many patients have a value for this observation
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() > 0:
            present = num.fillna(0) != 0
        else:
            present = s.notna() & (s.astype(str).str.strip() != "")
        count = int(present.sum())
        if count > 0:
            observation_data[col] = count
    
    if observation_data:
        sorted_data = dict(sorted(observation_data.items(), key=lambda x: x[1], reverse=True))
        
        # Keep FULL column names
        full_names = list(sorted_data.keys())
        
        # Create clean labels for display
        clean_labels = []
        for col_name in full_names:
            # Split by pipe and take last part for display
            if '|' in col_name:
                parts = col_name.split('|')
                clean_name = parts[-1].strip()
            else:
                clean_name = col_name
            # Truncate if too long
            if len(clean_name) > 50:
                clean_name = clean_name[:47] + "..."
            clean_labels.append(clean_name)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=clean_labels,  # Display names
            y=list(sorted_data.values()),
            marker_color='steelblue',
            text=list(sorted_data.values()),
            textposition='outside',
            customdata=full_names,  # Store full names for hover
            hovertemplate='<b>%{customdata}</b><br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Observation Presence Counts (FHIR Observations)',
            xaxis_title='Observation Name',
            yaxis_title='Patients with Observation',
            xaxis_tickangle=-45,
            height=600,
            margin=dict(b=250, t=100, l=80, r=80),
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig, observation_data
    else:
        return None, {}


def plot_condition_data(processed_data, cond_names, cond_column_map):
    """Plot condition data analysis (FHIR Conditions - confirmed)"""
    
    condition_data = {}

    # Get all condition columns from the map
    cond_cols = sorted(set((cond_column_map or {}).values()))
    
    for col in cond_cols:
        if col not in processed_data.columns:
            continue
        values = pd.to_numeric(processed_data[col], errors="coerce").fillna(0)
        count = int((values > 0).sum())
        if count > 0:
            condition_data[col] = count
    
    if condition_data:
        sorted_data = dict(sorted(condition_data.items(), key=lambda x: x[1], reverse=True))
        
        # Keep FULL column names
        full_names = list(sorted_data.keys())
        
        # Create clean labels for display
        clean_labels = []
        for col_name in full_names:
            # Split by pipe and take last part for display
            if '|' in col_name:
                parts = col_name.split('|')
                clean_name = parts[-1].strip()
            else:
                clean_name = col_name
            # Truncate if too long
            if len(clean_name) > 50:
                clean_name = clean_name[:47] + "..."
            clean_labels.append(clean_name)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=clean_labels,  # Display names
            y=list(sorted_data.values()),
            marker_color='mediumseagreen',
            text=list(sorted_data.values()),
            textposition='outside',
            customdata=full_names,  # Store full names for hover
            hovertemplate='<b>%{customdata}</b><br>Patients: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Confirmed Condition Counts (FHIR Conditions)',
            xaxis_title='Condition Name',
            yaxis_title='Number of Patients',
            xaxis_tickangle=-45,
            height=600,
            margin=dict(b=250, t=100, l=80, r=80),
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig, condition_data
    else:
        return None, {}


def correlation_analysis_page(processed_data, obs_names, cond_names, obs_column_map=None, cond_column_map=None):
    """Main correlation analysis page"""
    
    # Build column maps if not provided (backward compatibility)
    if obs_column_map is None:
        obs_column_map = _build_obs_column_map(processed_data, obs_names)
    
    if cond_column_map is None:
        cond_column_map = _build_cond_column_map(processed_data, cond_names)
    
    st.markdown('<h2 class="section-header">🔗 Correlation Analysis</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Correlation Matrix", "👁️ Observations", "🏥 Conditions"])
    
    with tab1:
        st.markdown("""
        <div class="info-box">
        <strong>📊 Correlation Analysis:</strong> This analysis examines the relationships between medical observations and conditions. 
        Positive correlations indicate conditions that tend to occur together, while negative correlations suggest mutual exclusivity.
        </div>
        """, unsafe_allow_html=True)
        
        fig, cross_corr = plot_correlation(processed_data, obs_names, cond_names, obs_column_map, cond_column_map)
        
        if fig is not None and cross_corr is not None:
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<h3 class="section-header">Correlation Matrix Table</h3>', unsafe_allow_html=True)
            
            cross_corr_formatted = cross_corr.round(3)
            st.dataframe(cross_corr_formatted, use_container_width=True)
            
        else:
            st.error("Unable to generate correlation analysis. Please check your data.")
    
    with tab2:
        st.markdown('<h3 class="section-header">👁️ Observation Analysis</h3>', unsafe_allow_html=True)
        
        obs_fig, obs_data = plot_observation_data(processed_data, obs_names, obs_column_map)
        
        if obs_fig is not None:
            st.plotly_chart(obs_fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Observation Types", len(obs_data))
            
            with col2:
                st.metric("Total Observations", sum(obs_data.values()))
            
            with col3:
                if obs_data:
                    top_obs = max(obs_data, key=obs_data.get)
                    st.metric("Highest Count", f"{obs_data[top_obs]}")
            
            if obs_data:
                st.markdown('<h4>Top Observations</h4>', unsafe_allow_html=True)
                sorted_obs = sorted(obs_data.items(), key=lambda x: x[1], reverse=True)
                obs_df = pd.DataFrame(sorted_obs, columns=['Observation', 'Count'])
                st.dataframe(obs_df, use_container_width=True)
        else:
            st.info("No observation data found")
    
    with tab3:
        st.markdown('<h3 class="section-header">🏥 Condition Analysis</h3>', unsafe_allow_html=True)
        
        cond_fig, cond_data = plot_condition_data(processed_data, cond_names, cond_column_map)
        
        if cond_fig is not None:
            st.plotly_chart(cond_fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Condition Types", len(cond_data))
            
            with col2:
                st.metric("Total Confirmed Cases", sum(cond_data.values()))
            
            with col3:
                if cond_data:
                    top_cond = max(cond_data, key=cond_data.get)
                    st.metric("Most Common", f"{cond_data[top_cond]} patients")
            
            if cond_data:
                st.markdown('<h4>Top Conditions</h4>', unsafe_allow_html=True)
                sorted_cond = sorted(cond_data.items(), key=lambda x: x[1], reverse=True)
                cond_df = pd.DataFrame(sorted_cond, columns=['Condition', 'Patient Count'])
                st.dataframe(cond_df, use_container_width=True)
        else:
            st.info("No condition data found")