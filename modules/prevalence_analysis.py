import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce a series to numeric (0/1 etc). Non-numeric -> NaN."""
    return pd.to_numeric(s, errors="coerce")

def get_binary_columns(df):
    """Get binary columns (0/1 or True/False)"""
    binary_cols = []
    for col in df.columns:
        values = _to_numeric_series(df[col])
        if values.notna().sum() == 0:
            continue
        unique_vals = set(values.dropna().unique())
        if unique_vals <= {0, 1} or unique_vals <= {True, False} or unique_vals <= {0.0, 1.0}:
            binary_cols.append(col)
    return binary_cols

def calculate_prevalence(df, disease_cols=None, case_value=1):
    """Calculate prevalence for binary/disease columns"""
    if disease_cols is None:
        # Get binary columns (assuming these represent conditions/diseases)
        disease_cols = get_binary_columns(df)
        # Remove non-medical columns
        exclude_cols = ['active', 'patient_id', 'gender']
        disease_cols = [col for col in disease_cols if col not in exclude_cols]
    
    if not disease_cols:
        return None
    
    prevalence_data = []
    
    for col in disease_cols:
        values = _to_numeric_series(df[col])
        valid = values.notna()
        total_population = int(valid.sum())
        n_cases = int((values.loc[valid] == case_value).sum())
        
        if total_population > 0:
            prevalence_prop = n_cases / total_population
            prevalence_pct = prevalence_prop * 100
        else:
            prevalence_prop = float('nan')
            prevalence_pct = float('nan')
        
        prevalence_data.append({
            'Condition': col,
            'Cases': n_cases,
            'Total_Population': total_population,
            'Prevalence_Proportion': prevalence_prop,
            'Prevalence_Percentage': prevalence_pct
        })
    
    prevalence_df = pd.DataFrame(prevalence_data)
    # Ensure numeric dtype for downstream nlargest/sorting
    prevalence_df['Prevalence_Percentage'] = pd.to_numeric(prevalence_df['Prevalence_Percentage'], errors='coerce')
    prevalence_df['Prevalence_Proportion'] = pd.to_numeric(prevalence_df['Prevalence_Proportion'], errors='coerce')
    prevalence_df['Cases'] = pd.to_numeric(prevalence_df['Cases'], errors='coerce')
    prevalence_df['Total_Population'] = pd.to_numeric(prevalence_df['Total_Population'], errors='coerce')
    return prevalence_df

def patient_segmentation(df, groupby_col='gender', top_n=5):
    """Perform patient segmentation analysis"""
    
    if groupby_col not in df.columns:
        return None
    
    # Get binary/condition columns for analysis
    disease_cols = get_binary_columns(df)
    exclude_cols = ['active', 'patient_id', groupby_col]
    disease_cols = [col for col in disease_cols if col not in exclude_cols]
    
    if not disease_cols:
        return None
    
    # Remove rows with missing groupby values
    df_clean = df.dropna(subset=[groupby_col])
    
    if df_clean.empty:
        return None
    
    # Coerce binary columns to numeric to avoid dtype=object issues in mean/nlargest
    disease_numeric = df_clean[disease_cols].apply(_to_numeric_series).fillna(0)
    group_disease = disease_numeric.groupby(df_clean[groupby_col]).mean()
    
    results = {
        "group_prevalence": group_disease,
        "top_conditions": {},
        "bottom_conditions": {},
        "group_summary": []
    }
    
    # Calculate differences for each group
    for group in group_disease.index:
        other_groups_mean = group_disease.drop(index=group).mean()
        group_diff = group_disease.loc[group] - other_groups_mean
        
        top_conditions = group_diff.nlargest(top_n)
        bottom_conditions = group_diff.nsmallest(top_n)
        
        results["top_conditions"][group] = [
            {
                'condition': cond, 
                'group_prevalence': group_disease.loc[group, cond],
                'others_prevalence': other_groups_mean[cond],
                'difference': diff
            }
            for cond, diff in top_conditions.items()
        ]
        
        results["bottom_conditions"][group] = [
            {
                'condition': cond, 
                'group_prevalence': group_disease.loc[group, cond],
                'others_prevalence': other_groups_mean[cond],
                'difference': diff
            }
            for cond, diff in bottom_conditions.items()
        ]
        
        # Group summary
        results["group_summary"].append({
            'Group': group,
            'Count': len(df_clean[df_clean[groupby_col] == group]),
            'Avg_Conditions': group_disease.loc[group].mean(),
            'Total_Conditions': (group_disease.loc[group] > 0).sum()
        })
    
    return results

def prevalence_analysis_page(processed_data):
    """Main prevalence analysis page"""
    
    st.markdown('<h2 class="section-header">📊 Prevalence Analysis</h2>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
    <strong>📊 Prevalence Analysis:</strong> This analysis calculates the prevalence of medical conditions 
    and performs patient segmentation to identify patterns across different demographic groups.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tab1, tab2 = st.tabs(["Condition Prevalence", "Patient Segmentation"])
    
    with tab1:
        st.markdown('<h3 class="section-header">Condition Prevalence Analysis</h3>', unsafe_allow_html=True)
        
        prevalence_result = calculate_prevalence(processed_data)
        
        if prevalence_result is not None:
            # Sort by prevalence percentage
            prevalence_sorted = prevalence_result.sort_values('Prevalence_Percentage', ascending=False)
            
            # Display table
            st.dataframe(prevalence_sorted, use_container_width=True)
            
            # Visualization
            top_conditions = prevalence_sorted.nlargest(15, 'Prevalence_Percentage')
            
            fig = px.bar(
                top_conditions,
                x='Prevalence_Percentage',
                y='Condition',
                orientation='h',
                title='Top 15 Conditions by Prevalence (%)',
                labels={'Prevalence_Percentage': 'Prevalence (%)', 'Condition': 'Medical Condition'}
            )
            
            # Clean condition names for better display
            clean_names = []
            for condition in top_conditions['Condition']:
                if '|' in condition:
                    clean_name = condition.split('|')[-1].strip()
                else:
                    clean_name = condition
                if len(clean_name) > 30:
                    clean_name = clean_name[:27] + "..."
                clean_names.append(clean_name)
            
            fig.update_layout(
                yaxis=dict(ticktext=clean_names, tickvals=top_conditions['Condition']),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            
        else:
            st.info("No suitable columns found for prevalence calculation.")
    
    with tab2:
        st.markdown('<h3 class="section-header">Patient Segmentation Analysis</h3>', unsafe_allow_html=True)
        
        # Segmentation options
        segmentation_col = st.selectbox(
            "Select segmentation variable:",
            ['gender', 'active'],
            help="Choose the variable to segment patients by"
        )
        
        segmentation_results = patient_segmentation(processed_data, groupby_col=segmentation_col)
        
        if segmentation_results is not None:
            # Display group summary
            summary_df = pd.DataFrame(segmentation_results["group_summary"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Group Summary:**")
                st.dataframe(summary_df, use_container_width=True)
            
            with col2:
                st.markdown("**Group Distribution:**")
                fig_pie = px.pie(
                    summary_df,
                    values='Count',
                    names='Group',
                    title=f'Distribution of Groups by {segmentation_col.title()}'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Group prevalences heatmap
            st.markdown('<h4>Condition Prevalence by Group</h4>', unsafe_allow_html=True)
            
            # Get top conditions overall
            top_conditions_overall = segmentation_results["group_prevalence"].mean().nlargest(10).index
            
            # Create heatmap
            heatmap_data = segmentation_results["group_prevalence"][top_conditions_overall].T
            
            # Clean condition names
            clean_condition_names = []
            for condition in heatmap_data.index:
                if '|' in condition:
                    clean_name = condition.split('|')[-1].strip()
                else:
                    clean_name = condition
                if len(clean_name) > 25:
                    clean_name = clean_name[:22] + "..."
                clean_condition_names.append(clean_name)
            
            heatmap_data.index = clean_condition_names
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='YlOrRd',
                text=np.round(heatmap_data.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig_heatmap.update_layout(
                title=f'Condition Prevalence Heatmap by {segmentation_col.title()}',
                xaxis_title=f'{segmentation_col.title()}',
                yaxis_title='Conditions',
                height=500
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Top differentiating conditions for each group
            st.markdown('<h4>Most Distinctive Conditions by Group</h4>', unsafe_allow_html=True)
            
            # Create tabs for each group
            group_tabs = st.tabs([f"{group.title()}" for group in segmentation_results["top_conditions"].keys()])
            
            for i, (group, tab) in enumerate(zip(segmentation_results["top_conditions"].keys(), group_tabs)):
                with tab:
                    group_df = pd.DataFrame(segmentation_results["top_conditions"][group])
                    
                    # Clean condition names
                    clean_conditions = []
                    for condition in group_df['condition']:
                        if '|' in condition:
                            clean_name = condition.split('|')[-1].strip()
                        else:
                            clean_name = condition
                        clean_conditions.append(clean_name)
                    
                    group_df['Clean_Condition'] = clean_conditions
                    
                    # Display table
                    display_df = group_df[['Clean_Condition', 'group_prevalence', 'others_prevalence', 'difference']].copy()
                    display_df.columns = ['Condition', 'Group Prevalence', 'Others Prevalence', 'Difference']
                    display_df['Group Prevalence'] = display_df['Group Prevalence'].round(3)
                    display_df['Others Prevalence'] = display_df['Others Prevalence'].round(3)
                    display_df['Difference'] = display_df['Difference'].round(3)
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Visualization
                    fig_group = go.Figure()
                    
                    fig_group.add_trace(go.Bar(
                        name=f'{group.title()} Group',
                        x=display_df['Condition'],
                        y=display_df['Group Prevalence'],
                        marker_color='lightblue'
                    ))
                    
                    fig_group.add_trace(go.Bar(
                        name='Other Groups',
                        x=display_df['Condition'],
                        y=display_df['Others Prevalence'],
                        marker_color='lightcoral'
                    ))
                    
                    fig_group.update_layout(
                        title=f'Condition Prevalence: {group.title()} vs Others',
                        xaxis_title='Conditions',
                        yaxis_title='Prevalence',
                        barmode='group',
                        height=400
                    )
                    fig_group.update_xaxes(tickangle=-45)
                    
                    st.plotly_chart(fig_group, use_container_width=True)
            
        else:
            st.info(f"Could not perform patient segmentation analysis by {segmentation_col}.")
