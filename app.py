import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr
import warnings
from mlxtend.frequent_patterns import apriori, association_rules
import os
warnings.filterwarnings('ignore')

# Import analysis modules
from modules.correlation_analysis import correlation_analysis_page
from modules.clustering_analysis import clustering_analysis_page
from modules.anomaly_detection import anomaly_detection_page
from modules.phenotyping_analysis import phenotyping_analysis_page
from modules.statistical_analysis import statistical_analysis_page
from modules.prevalence_analysis import prevalence_analysis_page
from modules.patient_network_analysis import symptom_pattern_analysis_page
from patient_data import FHIRData, PatientRepository
from dataframe import ObservationRepository, ConditionRepository, PatientDataProcessor

# Page configuration
st.set_page_config(
    page_title="Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium minimalistic styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 300;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }
    
    .section-header {
        font-size: 1.6rem;
        font-weight: 400;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    
    .info-box {
        background-color: #f9fafb;
        border-left: 3px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 6px;
        color: #374151;
        font-size: 0.9rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding-left: 16px;
        padding-right: 16px;
        background-color: #f3f4f6;
        border-radius: 6px 6px 0px 0px;
        color: #6b7280;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #1f2937;
        border-bottom: 2px solid #3b82f6;
    }
    
    .plot-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #ffffff;
        color: #374151;
        border: 1px solid #d1d5db;
        border-radius: 6px;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Data table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Scrollable content */
    .scrollable-content {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Environment variables with defaults
FHIR_BASE_URL = os.getenv("FHIR_BASE_URL", "http://65.0.127.208:30007/fhir")
DEFAULT_GROUP_ID = os.getenv("DEFAULT_GROUP_ID", "Lepto")

def _get_query_params():
    """Get query parameters in a version-compatible way"""
    try:
        # Streamlit >= 1.30
        return dict(st.query_params)
    except Exception:
        # Fallback for older versions
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}

def _get_group_id_from_url():
    """Extract group ID from URL query parameters"""
    params = _get_query_params()
    group_id = None
    if isinstance(params, dict):
        if 'databankId' in params:
            value = params['databankId']
            group_id = value[0] if isinstance(value, list) else value
        elif 'groupId' in params:
            value = params['groupId']
            group_id = value[0] if isinstance(value, list) else value
    return (group_id or "").strip() or None

def _set_query_params(params_dict):
    """Set query parameters in a version-compatible way"""
    try:
        # Streamlit >= 1.30
        st.query_params.clear()
        for k, v in params_dict.items():
            st.query_params[k] = v
    except Exception:
        # Fallback for older versions
        try:
            st.experimental_set_query_params(**params_dict)
        except Exception:
            pass

def _ensure_default_group_in_url(default_group: str = None):
    """Ensure a default group ID is in the URL if none is present"""
    if default_group is None:
        default_group = DEFAULT_GROUP_ID
    
    current = _get_group_id_from_url()
    if current:
        return current
    
    # Set default group in URL
    _set_query_params({"databankId": default_group})
    st.rerun()

# Load data from FHIR
@st.cache_data(show_spinner=True, ttl=3600)
def load_data_from_fhir(group_id: str):
    """Fetch patients and processed data from FHIR for a specific group id."""
    try:
        fhir = FHIRData(FHIR_BASE_URL, group_id)
        repo = PatientRepository(fhir)
        patients_df = repo.get_patients_dataframe()

        if patients_df.empty:
            raise ValueError(f"No patients found for group ID: {group_id}")

        observation_repo = ObservationRepository(f"{FHIR_BASE_URL}/Observation")
        condition_repo = ConditionRepository(f"{FHIR_BASE_URL}/Condition")
        processor = PatientDataProcessor(observation_repo, condition_repo, patients_df)
        processed_data = processor.process_patient_data()
        obs_names = processor.get_observation_names()
        cond_names = processor.get_condition_names()
        
        return processed_data, patients_df, obs_names, cond_names
    except Exception as e:
        raise Exception(f"Error loading data from FHIR: {str(e)}")

# Utility functions
def get_numeric_columns(df, exclude_binary=True):
    """Get numeric columns, optionally excluding binary columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude_binary:
        binary_like = [col for col in numeric_cols 
                      if set(df[col].dropna().unique()) <= {0, 1}]
        numeric_cols = [col for col in numeric_cols if col not in binary_like]
    return numeric_cols

def get_categorical_columns(df):
    """Get categorical columns"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def get_binary_columns(df):
    """Get binary columns (0/1 or True/False)"""
    binary_cols = []
    for col in df.columns:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals <= {0, 1} or unique_vals <= {True, False} or unique_vals <= {0.0, 1.0}:
            binary_cols.append(col)
    return binary_cols

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">Healthcare Analytics</h1>', unsafe_allow_html=True)
    
    # Ensure group ID is in URL
    group_id = _ensure_default_group_in_url()
    
    # Display current configuration
    with st.sidebar.expander("📋 Configuration", expanded=False):
        st.write(f"**FHIR Base URL:** {FHIR_BASE_URL}")
        st.write(f"**Group ID:** {group_id}")
        st.write(f"**Default Group:** {DEFAULT_GROUP_ID}")

    # Load data from FHIR
    with st.spinner(f"Loading data from FHIR for group '{group_id}'..."):
        try:
            processed_data, patients_df, obs_names, cond_names = load_data_from_fhir(group_id)
        except Exception as e:
            st.error(f"Failed to load data from FHIR for group '{group_id}': {e}")
            st.info("Please check:")
            st.markdown("""
            - FHIR server is accessible
            - Group ID is correct
            - Environment variables are set correctly:
              - `FHIR_BASE_URL` (default: http://65.0.127.208:30007/fhir)
              - `DEFAULT_GROUP_ID` (default: Lepto)
            """)
            return
    
    # Sidebar navigation with buttons
    st.sidebar.title("Operations Panel")
    st.sidebar.markdown("---")
    
    # Navigation buttons
    pages = {
        "Data Overview": "Data Overview",
        "Correlation": "Correlation Analysis", 
        "Patient Clustering": "Patient Clustering",
        "Anomaly Detection": "Anomaly Detection",
        "Patient Phenotyping": "Patient Phenotyping",
        "Statistical Analysis": "Statistical Analysis",
        "Prevalence Analysis": "Prevalence Analysis",
        "Patient Network": "Patient Network Analysis"
    }
    
    # Initialize session state for page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Data Overview"
    
    # Create navigation buttons
    for page_name, page_value in pages.items():
        if st.sidebar.button(page_name, key=f"nav_{page_value}", use_container_width=True):
            st.session_state.current_page = page_value
    
    st.sidebar.markdown("---")
    
    # Data info in sidebar
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.metric("Patients", len(processed_data))
    st.sidebar.metric("Features", len(processed_data.columns))
    st.sidebar.metric("Observations", len(obs_names))
    st.sidebar.metric("Conditions", len(cond_names))
    
    page = st.session_state.current_page
    
    # Data Overview Page
    if page == "Data Overview":
        st.markdown('<h2 class="section-header">Data Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(processed_data))
        
        with col2:
            st.metric("Total Features", len(processed_data.columns))
        
        with col3:
            st.metric("Observations", len(obs_names))
        
        with col4:
            st.metric("Conditions", len(cond_names))
        
        # Data tables with pagination
        st.markdown('<h3 class="section-header">Patient Data</h3>', unsafe_allow_html=True)
        
        # Pagination for patients table
        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
        total_pages = len(patients_df) // page_size + (1 if len(patients_df) % page_size > 0 else 0)
        
        if total_pages > 1:
            page_num = st.selectbox("Page", range(1, total_pages + 1))
            start_idx = (page_num - 1) * page_size
            end_idx = start_idx + page_size
            st.dataframe(patients_df.iloc[start_idx:end_idx], use_container_width=True)
        else:
            st.dataframe(patients_df, use_container_width=True)
        
        # Processed data table with scrollable content
        st.markdown('<h3 class="section-header">Processed Medical Data</h3>', unsafe_allow_html=True)
        
        # Show all columns in a scrollable container
        st.markdown('<div class="scrollable-content">', unsafe_allow_html=True)
        st.dataframe(processed_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data summary
        st.markdown('<h3 class="section-header">Data Summary</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Types:**")
            dtype_counts = processed_data.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count} columns")
        
        with col2:
            st.markdown("**Missing Values:**")
            missing_data = processed_data.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                for col, missing in missing_data.items():
                    st.write(f"- {col}: {missing} missing")
            else:
                st.write("No missing values found")
    
    # Correlation Analysis Page
    elif page == "Correlation Analysis":
        correlation_analysis_page(processed_data, obs_names, cond_names)
    
    # Patient Clustering Page
    elif page == "Patient Clustering":
        clustering_analysis_page(processed_data)
    
    # Anomaly Detection Page
    elif page == "Anomaly Detection":
        anomaly_detection_page(processed_data)
    
    # Patient Phenotyping Page
    elif page == "Patient Phenotyping":
        phenotyping_analysis_page(processed_data)
    
    # Statistical Analysis Page
    elif page == "Statistical Analysis":
        statistical_analysis_page(processed_data)
    
    # Prevalence Analysis Page
    elif page == "Prevalence Analysis":
        prevalence_analysis_page(processed_data)
    
    # Patient Network Analysis Page
    elif page == "Patient Network Analysis":
        # Pass FHIR-derived names so observations/conditions are classified correctly
        symptom_pattern_analysis_page(processed_data, obs_names, cond_names)
    

if __name__ == "__main__":
    main()