import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr
import streamlit as st

def get_numeric_columns(df, exclude_binary=True):
    """Get numeric columns, optionally excluding binary columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude_binary:
        binary_like = [col for col in numeric_cols 
                      if set(df[col].dropna().unique()) <= {0, 1}]
        numeric_cols = [col for col in numeric_cols if col not in binary_like]
    return numeric_cols

def get_binary_columns(df):
    """Get binary columns (0/1 or True/False)"""
    binary_cols = []
    for col in df.columns:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals <= {0, 1} or unique_vals <= {True, False} or unique_vals <= {0.0, 1.0}:
            binary_cols.append(col)
    return binary_cols

def calculate_means(df):
    """Calculate means for numeric columns"""
    numeric_cols = get_numeric_columns(df, exclude_binary=True)
    
    if not numeric_cols:
        return None
    
    means_data = []
    for col in numeric_cols:
        means_data.append({
            'Column': col,
            'Mean': df[col].mean(),
            'Count': df[col].count(),
            'Missing': df[col].isnull().sum()
        })
    
    means_df = pd.DataFrame(means_data)
    return means_df

def calculate_medians(df):
    """Calculate medians for numeric columns"""
    numeric_cols = get_numeric_columns(df, exclude_binary=True)
    
    if not numeric_cols:
        return None
    
    medians_data = []
    for col in numeric_cols:
        medians_data.append({
            'Column': col,
            'Median': df[col].median(),
            'Q1': df[col].quantile(0.25),
            'Q3': df[col].quantile(0.75),
            'IQR': df[col].quantile(0.75) - df[col].quantile(0.25)
        })
    
    medians_df = pd.DataFrame(medians_data)
    return medians_df

def get_mode_info(series):
    """Get mode information for a series"""
    mode_vals = series.mode()
    if mode_vals.empty:
        return {"mode": None, "count": 0}
    elif len(mode_vals) == 1:
        return {"mode": mode_vals.iloc[0], "count": 1}
    else:
        return {"mode": mode_vals.tolist(), "count": len(mode_vals)}

def calculate_modes(df):
    """Calculate modes for numeric columns"""
    numeric_cols = get_numeric_columns(df, exclude_binary=True)
    
    if not numeric_cols:
        return None
    
    modes_data = []
    for col in numeric_cols:
        mode_info = get_mode_info(df[col])
        modes_data.append({
            'Column': col,
            'Mode': mode_info['mode'],
            'Mode_Count': mode_info['count'],
            'Unique_Values': df[col].nunique(),
            'Most_Frequent_Count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
        })
    
    modes_df = pd.DataFrame(modes_data)
    return modes_df

def calculate_std(df):
    """Calculate standard deviation for numeric columns"""
    numeric_cols = get_numeric_columns(df, exclude_binary=True)
    
    if not numeric_cols:
        return None
    
    std_data = []
    for col in numeric_cols:
        std_data.append({
            'Column': col,
            'Std_Deviation': df[col].std(),
            'Variance': df[col].var(),
            'Mean': df[col].mean(),
            'CV_Percent': (df[col].std() / df[col].mean() * 100) if df[col].mean() != 0 else np.nan
        })
    
    std_df = pd.DataFrame(std_data)
    return std_df

def calculate_ranges(df):
    """Calculate range for numeric columns"""
    numeric_cols = get_numeric_columns(df, exclude_binary=True)
    
    if not numeric_cols:
        return None
    
    range_data = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            range_data.append({
                'Column': col,
                'Range': col_data.max() - col_data.min(),
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Span': f"{col_data.min():.2f} to {col_data.max():.2f}"
            })
    
    range_df = pd.DataFrame(range_data)
    return range_df

def calculate_correlations(df):
    """Calculate correlation coefficients for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    correlation_results = []
    
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            
            # Remove NaN values
            data = df[[col1, col2]].dropna()
            
            if len(data) > 1:
                # Calculate correlations
                pearson_r, pearson_p = pearsonr(data[col1], data[col2])
                spearman_r, spearman_p = spearmanr(data[col1], data[col2])
                
                correlation_results.append({
                    'Column_1': col1,
                    'Column_2': col2,
                    'Pearson_Coefficient': round(pearson_r, 6),
                    'Pearson_P_Value': round(pearson_p, 6),
                    'Spearman_Coefficient': round(spearman_r, 6),
                    'Spearman_P_Value': round(spearman_p, 6)
                })
    
    correlation_df = pd.DataFrame(correlation_results)
    return correlation_df

def calculate_covariances(df):
    """Calculate covariance for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    covariance_results = []
    
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            
            # Remove NaN values
            data = df[[col1, col2]].dropna()
            
            if len(data) > 1:
                # Calculate covariance
                cov_value = data[col1].cov(data[col2])
                
                covariance_results.append({
                    'Column_1': col1,
                    'Column_2': col2,
                    'Covariance': round(cov_value, 6)
                })
    
    covariance_df = pd.DataFrame(covariance_results)
    return covariance_df

def statistical_analysis_page(processed_data):
    """Main statistical analysis page"""
    
    st.markdown('<h2 class="section-header">📈 Statistical Analysis</h2>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
    <strong>📈 Statistical Analysis:</strong> This comprehensive analysis provides descriptive statistics, 
    correlation analysis, and covariance calculations for all numeric variables in the dataset.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different statistical analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Descriptive Stats", "📈 Means", "📉 Medians", "🎯 Modes", 
        "📏 Standard Deviation", "📐 Ranges", "🔗 Correlations"
    ])
    
    with tab1:
        st.markdown('<h3 class="section-header">Descriptive Statistics</h3>', unsafe_allow_html=True)
        
        numeric_cols = get_numeric_columns(processed_data, exclude_binary=True)
        
        if numeric_cols:
            desc_stats = processed_data[numeric_cols].describe()
            st.dataframe(desc_stats, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Numeric Columns", len(numeric_cols))
            
            with col2:
                st.metric("Total Observations", len(processed_data))
            
            with col3:
                st.metric("Missing Values", processed_data[numeric_cols].isnull().sum().sum())
            
            with col4:
                st.metric("Complete Cases", len(processed_data[numeric_cols].dropna()))
        else:
            st.info("No numeric columns found for descriptive statistics.")
    
    with tab2:
        st.markdown('<h3 class="section-header">Mean Analysis</h3>', unsafe_allow_html=True)
        
        means_result = calculate_means(processed_data)
        
        if means_result is not None:
            st.dataframe(means_result, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                means_result, 
                x='Column', 
                y='Mean',
                title='Mean Values of Numeric Columns',
                labels={'Column': 'Column Name', 'Mean': 'Mean Value'},
                color='Mean',
                color_continuous_scale='Blues'
            )
            
            # Clean column names for display
            clean_names = []
            for col in means_result['Column']:
                if len(col) > 15:
                    clean_name = col[:12] + "..."
                else:
                    clean_name = col
                clean_names.append(clean_name)
            
            fig.update_layout(
                xaxis=dict(ticktext=clean_names, tickvals=means_result['Column']),
                height=500
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for mean calculation.")
    
    with tab3:
        st.markdown('<h3 class="section-header">Median Analysis</h3>', unsafe_allow_html=True)
        
        medians_result = calculate_medians(processed_data)
        
        if medians_result is not None:
            st.dataframe(medians_result, use_container_width=True)
            
            # Box plot visualization
            numeric_cols = get_numeric_columns(processed_data, exclude_binary=True)
            if len(numeric_cols) > 0:
                cols_to_plot = numeric_cols[:5]  # Show first 5 columns
                
                fig = go.Figure()
                for col in cols_to_plot:
                    clean_name = col[:20] + "..." if len(col) > 20 else col
                    fig.add_trace(go.Box(
                        y=processed_data[col].dropna(),
                        name=clean_name,
                        boxpoints='outliers',
                        jitter=0.3,
                        pointpos=-1.8
                    ))
                
                fig.update_layout(
                    title='Box Plot of Numeric Columns (showing median, quartiles)',
                    yaxis_title='Values',
                    height=500,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Median comparison plot
                fig_median = px.bar(
                    medians_result,
                    x='Column',
                    y='Median',
                    title='Median Values Comparison',
                    color='Median',
                    color_continuous_scale='Viridis'
                )
                
                # Clean column names
                clean_names = []
                for col in medians_result['Column']:
                    if len(col) > 15:
                        clean_name = col[:12] + "..."
                    else:
                        clean_name = col
                    clean_names.append(clean_name)
                
                fig_median.update_layout(
                    xaxis=dict(ticktext=clean_names, tickvals=medians_result['Column']),
                    height=400
                )
                fig_median.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_median, use_container_width=True)
        else:
            st.info("No numeric columns available for median calculation.")
    
    with tab4:
        st.markdown('<h3 class="section-header">Mode Analysis</h3>', unsafe_allow_html=True)
        
        modes_result = calculate_modes(processed_data)
        
        if modes_result is not None:
            st.dataframe(modes_result, use_container_width=True)
            
            # Mode visualization
            fig_mode = px.bar(
                modes_result,
                x='Column',
                y='Most_Frequent_Count',
                title='Most Frequent Value Counts by Column',
                color='Most_Frequent_Count',
                color_continuous_scale='Oranges'
            )
            
            # Clean column names
            clean_names = []
            for col in modes_result['Column']:
                if len(col) > 15:
                    clean_name = col[:12] + "..."
                else:
                    clean_name = col
                clean_names.append(clean_name)
            
            fig_mode.update_layout(
                xaxis=dict(ticktext=clean_names, tickvals=modes_result['Column']),
                height=400
            )
            fig_mode.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_mode, use_container_width=True)
            
            # Unique values distribution
            fig_unique = px.bar(
                modes_result,
                x='Column',
                y='Unique_Values',
                title='Number of Unique Values by Column',
                color='Unique_Values',
                color_continuous_scale='Greens'
            )
            
            fig_unique.update_layout(
                xaxis=dict(ticktext=clean_names, tickvals=modes_result['Column']),
                height=400
            )
            fig_unique.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_unique, use_container_width=True)
        else:
            st.info("No numeric columns available for mode calculation.")
    
    with tab5:
        st.markdown('<h3 class="section-header">Standard Deviation Analysis</h3>', unsafe_allow_html=True)
        
        std_result = calculate_std(processed_data)
        
        if std_result is not None:
            st.dataframe(std_result, use_container_width=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(
                    std_result, 
                    x='Column', 
                    y='Std_Deviation',
                    title='Standard Deviation of Numeric Columns',
                    color='Std_Deviation',
                    color_continuous_scale='Reds'
                )
                
                # Clean column names
                clean_names = []
                for col in std_result['Column']:
                    if len(col) > 15:
                        clean_name = col[:12] + "..."
                    else:
                        clean_name = col
                    clean_names.append(clean_name)
                
                fig1.update_layout(
                    xaxis=dict(ticktext=clean_names, tickvals=std_result['Column']),
                    height=400
                )
                fig1.update_xaxes(tickangle=-45)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                cv_data = std_result.dropna(subset=['CV_Percent'])
                if len(cv_data) > 0:
                    fig2 = px.bar(
                        cv_data, 
                        x='Column', 
                        y='CV_Percent',
                        title='Coefficient of Variation (%)',
                        color='CV_Percent',
                        color_continuous_scale='Purples'
                    )
                    
                    # Clean column names for CV plot
                    clean_names_cv = []
                    for col in cv_data['Column']:
                        if len(col) > 15:
                            clean_name = col[:12] + "..."
                        else:
                            clean_name = col
                        clean_names_cv.append(clean_name)
                    
                    fig2.update_layout(
                        xaxis=dict(ticktext=clean_names_cv, tickvals=cv_data['Column']),
                        height=400
                    )
                    fig2.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No numeric columns available for standard deviation calculation.")
    
    with tab6:
        st.markdown('<h3 class="section-header">Range Analysis</h3>', unsafe_allow_html=True)
        
        range_result = calculate_ranges(processed_data)
        
        if range_result is not None:
            st.dataframe(range_result, use_container_width=True)
            
            # Visualization
            fig = go.Figure()
            
            # Clean column names
            clean_names = []
            for col in range_result['Column']:
                if len(col) > 15:
                    clean_name = col[:12] + "..."
                else:
                    clean_name = col
                clean_names.append(clean_name)
            
            fig.add_trace(go.Bar(
                name='Min',
                x=range_result['Column'],
                y=range_result['Min'],
                marker_color='lightblue',
                text=range_result['Min'].round(2),
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                name='Max',
                x=range_result['Column'],
                y=range_result['Max'],
                marker_color='darkblue',
                text=range_result['Max'].round(2),
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Min vs Max Values',
                xaxis_title='Columns',
                yaxis_title='Values',
                barmode='group',
                height=500,
                xaxis=dict(ticktext=clean_names, tickvals=range_result['Column'])
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Range visualization
            fig_range = px.bar(
                range_result,
                x='Column',
                y='Range',
                title='Range (Max - Min) by Column',
                color='Range',
                color_continuous_scale='Blues'
            )
            
            fig_range.update_layout(
                xaxis=dict(ticktext=clean_names, tickvals=range_result['Column']),
                height=400
            )
            fig_range.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_range, use_container_width=True)
        else:
            st.info("No numeric columns available for range calculation.")
    
    with tab7:
        st.markdown('<h3 class="section-header">Correlation Analysis</h3>', unsafe_allow_html=True)
        
        # Correlation coefficients
        correlation_df = calculate_correlations(processed_data)
        
        if len(correlation_df) > 0:
            st.markdown("**Correlation Coefficients:**")
            
            # Show top correlations
            correlation_df['Abs_Pearson'] = abs(correlation_df['Pearson_Coefficient'])
            top_correlations = correlation_df.nlargest(10, 'Abs_Pearson')
            
            st.dataframe(top_correlations[['Column_1', 'Column_2', 'Pearson_Coefficient', 'Spearman_Coefficient']], 
                        use_container_width=True)
            
            
            # Covariance analysis
            st.markdown("**Covariance Analysis:**")
            covariance_df = calculate_covariances(processed_data)
            
            if len(covariance_df) > 0:
                covariance_df['Abs_Covariance'] = abs(covariance_df['Covariance'])
                top_covariances = covariance_df.nlargest(10, 'Abs_Covariance')
                st.dataframe(top_covariances[['Column_1', 'Column_2', 'Covariance']], 
                            use_container_width=True)
        else:
            st.info("No numeric columns found for correlation analysis.")
