"""
Symptom Pattern Analysis Module
Wrapper for easy integration with Streamlit app
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
from datetime import datetime
import streamlit.components.v1 as components
from typing import List, Optional, Dict, Tuple
from collections import Counter, defaultdict
from pydantic import BaseModel, Field, field_validator
from scipy.stats import chi2_contingency
from itertools import combinations
import pickle
import json


def convert_to_serializable(obj):
    """Convert numpy types and other non-serializable types to native Python types"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj


class SymptomPatternParams(BaseModel):
    """Parameters for symptom pattern analysis"""
    min_prevalence: float = Field(ge=0.0, le=1.0, default=0.05)
    min_co_occurrence: int = Field(ge=2, default=3)
    max_pattern_size: int = Field(ge=2, le=5, default=3)
    min_correlation: float = Field(ge=0.0, le=1.0, default=0.3)
    exclude_cols: Optional[List[str]] = []
    calculate_statistics: bool = True


class SymptomPatternAnalyzer:
    """Comprehensive symptom pattern analyzer with network visualization"""
    
    def __init__(self, df: pd.DataFrame, obs_names: List[str] = None, cond_names: List[str] = None):
        self.df = df
        # Store the FULL observation and condition names from FHIR (with pipes)
        self.observation_names = set(obs_names) if obs_names else set()
        self.condition_names = set(cond_names) if cond_names else set()
        self.G = None
        
    def _identify_symptom_columns(self, params: SymptomPatternParams) -> List[str]:
        """
        Identify valid symptom/condition columns based on FHIR-provided names.
        
        The key insight: 
        - obs_names contains full piped names like "House (environment) | Count of entities..."
        - cond_names contains condition names like "Eruption of skin"
        - DataFrame columns match these exactly, OR have numbered suffixes like " (2)", " (3)"
        """
        symptom_cols = []
        exclude_base = {'patient_id', 'last_updated', 'gender', 'active'}
        
        for col in self.df.columns:
            if col in exclude_base or col in params.exclude_cols:
                continue
            
            # Direct match with observation names
            if col in self.observation_names:
                symptom_cols.append(col)
                continue
            
            # Direct match with condition names
            if col in self.condition_names:
                symptom_cols.append(col)
                continue
            
            # Check if this is a numbered variant of an observation
            # e.g., "House (environment) | Count... (2)" matches base "House (environment) | Count..."
            is_numbered_obs = False
            for obs_name in self.observation_names:
                # Check if col starts with obs_name and ends with " (N)"
                if col.startswith(obs_name + " (") and col[len(obs_name)+2:-1].isdigit():
                    symptom_cols.append(col)
                    is_numbered_obs = True
                    break
            
            if is_numbered_obs:
                continue
            
            # Fallback: check if it's a binary column with actual data
            # (This helps catch edge cases where names might not match perfectly)
            if self.df[col].dtype in ['int64', 'float64', 'bool', 'Int64', 'Float64']:
                non_null_vals = self.df[col].dropna()
                if len(non_null_vals) > 0:
                    unique_vals = non_null_vals.unique()
                    if len(unique_vals) <= 2 and all(v in [0, 1, True, False, 0.0, 1.0] for v in unique_vals):
                        # Only add if we have some positive cases
                        if (non_null_vals == 1).sum() > 0 or (non_null_vals == True).sum() > 0:
                            symptom_cols.append(col)
        
        return symptom_cols

    def _calculate_symptom_prevalence(self, symptoms_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate prevalence (frequency) of each symptom"""
        total_patients = len(symptoms_df)
        prevalence = {}
        
        for col in symptoms_df.columns:
            count = int(symptoms_df[col].sum())
            prevalence[col] = float(count / total_patients if total_patients > 0 else 0)
        
        return prevalence

    def _find_co_occurring_patterns(self, symptoms_df: pd.DataFrame, 
                                   params: SymptomPatternParams,
                                   prevalence: Dict[str, float]) -> List[Dict]:
        """Find patterns of co-occurring symptoms"""
        patterns = []
        
        valid_symptoms = [s for s, p in prevalence.items() if p >= params.min_prevalence]
        
        for pattern_size in range(2, min(params.max_pattern_size + 1, len(valid_symptoms) + 1)):
            for symptom_combo in combinations(valid_symptoms, pattern_size):
                mask = symptoms_df[list(symptom_combo)].all(axis=1)
                co_occurrence_count = int(mask.sum())
                
                if co_occurrence_count >= params.min_co_occurrence:
                    pattern_prevalence = float(co_occurrence_count / len(symptoms_df))
                    
                    patterns.append({
                        'symptoms': list(symptom_combo),
                        'pattern_size': int(pattern_size),
                        'co_occurrence_count': int(co_occurrence_count),
                        'pattern_prevalence': float(pattern_prevalence),
                        'affected_patients': int(co_occurrence_count)
                    })
        
        return patterns

    def _calculate_symptom_correlations(self, symptoms_df: pd.DataFrame) -> List[Dict]:
        """Calculate pairwise correlations between symptoms"""
        correlations = []
        symptoms = symptoms_df.columns.tolist()
        
        for i, symptom1 in enumerate(symptoms):
            for symptom2 in symptoms[i+1:]:
                corr_matrix = symptoms_df[[symptom1, symptom2]].corr()
                correlation = corr_matrix.loc[symptom1, symptom2]
                
                both_present = int((symptoms_df[symptom1] & symptoms_df[symptom2]).sum())
                
                if not np.isnan(correlation) and both_present > 0:
                    correlations.append({
                        'symptom_1': str(symptom1),
                        'symptom_2': str(symptom2),
                        'correlation': float(correlation),
                        'co_occurrence_count': int(both_present)
                    })
        
        return sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)

    def _calculate_chi_square_associations(self, symptoms_df: pd.DataFrame) -> List[Dict]:
        """Calculate chi-square test for symptom associations"""
        associations = []
        symptoms = symptoms_df.columns.tolist()
        
        for i, symptom1 in enumerate(symptoms):
            for symptom2 in symptoms[i+1:]:
                contingency = pd.crosstab(
                    symptoms_df[symptom1], 
                    symptoms_df[symptom2]
                )
                
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    
                    if p_value < 0.05:
                        associations.append({
                            'symptom_1': str(symptom1),
                            'symptom_2': str(symptom2),
                            'chi_square': float(chi2),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        })
                except Exception:
                    continue
        
        return sorted(associations, key=lambda x: x['p_value'])

    def _build_symptom_network(self, symptoms_df: pd.DataFrame, 
                               correlations: List[Dict],
                               min_correlation: float) -> nx.Graph:
        """Build network graph from symptom correlations"""
        G = nx.Graph()
        
        # Add nodes
        for col in symptoms_df.columns:
            prevalence = symptoms_df[col].mean()
            G.add_node(col, prevalence=float(prevalence))
        
        # Add edges
        for corr in correlations:
            if abs(corr['correlation']) >= min_correlation:
                G.add_edge(
                    corr['symptom_1'],
                    corr['symptom_2'],
                    weight=abs(float(corr['correlation'])),
                    correlation=float(corr['correlation']),
                    co_occurrence=int(corr['co_occurrence_count'])
                )
        
        return G

    def _calculate_network_statistics(self, G: nx.Graph) -> Dict:
        """Calculate network statistics"""
        if G.number_of_nodes() == 0:
            return {
                'nodes': 0,
                'edges': 0,
                'avg_degree': 0.0,
                'density': 0.0,
                'avg_clustering': 0.0,
                'connected_components': 0
            }
        
        return {
            'nodes': int(G.number_of_nodes()),
            'edges': int(G.number_of_edges()),
            'avg_degree': float(sum(dict(G.degree()).values()) / G.number_of_nodes()) if G.number_of_nodes() > 0 else 0.0,
            'density': float(nx.density(G)),
            'avg_clustering': float(nx.average_clustering(G)) if G.number_of_nodes() > 0 else 0.0,
            'connected_components': int(nx.number_connected_components(G))
        }

    def _create_interactive_pyvis_network(self, G: nx.Graph) -> Network:
        """Create interactive network visualization using pyvis"""
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#000000")
        
        # Configure physics for better layout
        net.force_atlas_2based(
            gravity=-50,
            central_gravity=0.01,
            spring_length=100,
            spring_strength=0.08,
            damping=0.4
        )
        
        # Add nodes with sizing based on degree
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        for node in G.nodes():
            prevalence = G.nodes[node].get('prevalence', 0)
            degree = degrees.get(node, 0)
            
            # Size based on degree
            size = 10 + (degree / max_degree) * 30
            
            # Color based on degree (blue to red gradient)
            color_intensity = int((degree / max_degree) * 255) if max_degree > 0 else 0
            color = f'#{color_intensity:02x}{100:02x}{255-color_intensity:02x}'
            
            # Create readable label (take last part after pipe if present)
            label = node.split('|')[-1].strip() if '|' in node else node
            if len(label) > 30:
                label = label[:27] + "..."
            
            net.add_node(
                node,
                label=label,
                title=f"{node}\nPrevalence: {prevalence:.2%}\nConnections: {degree}",
                size=size,
                color=color
            )
        
        # Add edges with coloring based on correlation
        for edge in G.edges(data=True):
            correlation = edge[2].get('correlation', 0)
            weight = edge[2].get('weight', 0)
            co_occurrence = edge[2].get('co_occurrence', 0)
            
            # Color: blue for positive, red for negative
            if correlation > 0:
                color = f'rgba(0, 0, 255, {min(1.0, weight)})'
            else:
                color = f'rgba(255, 0, 0, {min(1.0, weight)})'
            
            # Width based on absolute correlation
            width = 1 + abs(correlation) * 5
            
            net.add_edge(
                edge[0],
                edge[1],
                title=f"Correlation: {correlation:.3f}\nCo-occurrence: {co_occurrence}",
                color=color,
                width=width
            )
        
        return net

    def analyze(self, params: SymptomPatternParams) -> Dict:
        """
        Run complete symptom pattern analysis
        
        Returns a dictionary with all analysis results
        """
        # Identify symptom columns using FHIR-provided names
        symptom_cols = self._identify_symptom_columns(params)
        
        if len(symptom_cols) == 0:
            return {
                'error': 'No symptom columns found',
                'symptom_columns': [],
                'total_patients': len(self.df),
                'patients_with_symptoms': 0,
                'symptom_breakdown': {'observations': 0, 'conditions': 0}
            }
        
        # Create binary symptom dataframe
        symptoms_df = pd.DataFrame()
        for col in symptom_cols:
            # Convert to binary (presence/absence)
            values = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
            symptoms_df[col] = (values > 0).astype(int)
        
        # Calculate prevalence
        prevalence = self._calculate_symptom_prevalence(symptoms_df)
        
        # Find co-occurring patterns
        patterns = self._find_co_occurring_patterns(symptoms_df, params, prevalence)
        
        # Calculate correlations
        correlations = self._calculate_symptom_correlations(symptoms_df)
        
        # Filter by minimum correlation
        filtered_correlations = [c for c in correlations if abs(c['correlation']) >= params.min_correlation]
        
        # Build network
        self.G = self._build_symptom_network(symptoms_df, correlations, params.min_correlation)
        
        # Calculate statistics
        network_stats = self._calculate_network_statistics(self.G)
        
        # Chi-square associations (if enabled)
        chi_square_associations = []
        if params.calculate_statistics:
            chi_square_associations = self._calculate_chi_square_associations(symptoms_df)
        
        # Count observations vs conditions in symptom_cols
        obs_count = 0
        cond_count = 0
        for col in symptom_cols:
            # Check if col is an observation (exact match or numbered variant)
            is_obs = col in self.observation_names
            if not is_obs:
                # Check numbered variants
                for obs_name in self.observation_names:
                    if col.startswith(obs_name + " (") and col[len(obs_name)+2:-1].isdigit():
                        is_obs = True
                        break
            
            if is_obs:
                obs_count += 1
            elif col in self.condition_names:
                cond_count += 1
        
        # Prepare results
        results = {
            'symptom_columns': symptom_cols,
            'total_patients': int(len(self.df)),
            'patients_with_symptoms': int((symptoms_df.sum(axis=1) > 0).sum()),
            'symptom_breakdown': {
                'observations': obs_count,
                'conditions': cond_count
            },
            'symptom_prevalence': [
                {'symptom': k, 'prevalence': float(v), 'count': int(symptoms_df[k].sum())}
                for k, v in sorted(prevalence.items(), key=lambda x: x[1], reverse=True)
            ],
            'co_occurring_patterns': {
                'total_patterns': int(len(patterns)),
                'patterns': patterns
            },
            'symptom_correlations': {
                'total_pairs': int(len(correlations)),
                'filtered_pairs': int(len(filtered_correlations)),
                'top_correlations': filtered_correlations
            },
            'network_statistics': network_stats,
            'statistical_associations': {
                'total_significant': int(len(chi_square_associations)),
                'associations': chi_square_associations
            }
        }
        
        return convert_to_serializable(results)


def symptom_pattern_analysis_page(processed_data: pd.DataFrame, obs_names: List[str], cond_names: List[str]):
    """
    Streamlit page for symptom pattern analysis
    
    Args:
        processed_data: DataFrame with patient data
        obs_names: List of observation names from FHIR (full piped names)
        cond_names: List of condition names from FHIR
    """
    st.markdown('<h2 class="section-header">🕸️ Patient Network Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>🕸️ Network Analysis:</strong> This analysis identifies patterns in how symptoms and conditions co-occur across patients.
    The network graph shows correlations between different medical observations and conditions.
    </div>
    """, unsafe_allow_html=True)
    
    # Parameters
    with st.expander("⚙️ Analysis Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            min_prevalence = st.slider(
                "Minimum Symptom Prevalence",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.05,
                help="Minimum fraction of patients that must have a symptom for it to be included"
            )
            
            min_correlation = st.slider(
                "Minimum Correlation",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Minimum correlation strength to show in network"
            )
        
        with col2:
            min_co_occurrence = st.number_input(
                "Minimum Co-occurrence Count",
                min_value=2,
                max_value=20,
                value=3,
                help="Minimum number of patients for a pattern to be considered"
            )
            
            max_pattern_size = st.slider(
                "Maximum Pattern Size",
                min_value=2,
                max_value=5,
                value=3,
                help="Maximum number of symptoms in a co-occurrence pattern"
            )
    
    # Run analysis
    params = SymptomPatternParams(
        min_prevalence=min_prevalence,
        min_co_occurrence=min_co_occurrence,
        max_pattern_size=max_pattern_size,
        min_correlation=min_correlation,
        exclude_cols=['patient_id', 'gender', 'active', 'last_updated'],
        calculate_statistics=True
    )
    
    with st.spinner("Analyzing symptom patterns..."):
        analyzer = SymptomPatternAnalyzer(processed_data, obs_names, cond_names)
        results = analyzer.analyze(params)
    
    # Check for errors
    if 'error' in results:
        st.error(f"Analysis failed: {results['error']}")
        return
    
    # Display summary statistics
    stats = {
        'total_patients': results['total_patients'],
        'patients_with_symptoms': results['patients_with_symptoms'],
        'symptom_breakdown': results['symptom_breakdown']
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", stats['total_patients'])
    
    with col2:
        st.metric("Patients with Symptoms", stats['patients_with_symptoms'])
    
    with col3:
        st.metric("Observations", stats['symptom_breakdown']['observations'])
    
    with col4:
        st.metric("Conditions", stats['symptom_breakdown']['conditions'])
    
    # Key insights
    st.markdown(f"""
    <div class="info-box">
    <strong>Key Pattern Insights:</strong><br>
    • <strong>Total Patients Analyzed</strong>: {stats['total_patients']}<br>
    • <strong>Patients with Symptoms</strong>: {stats['patients_with_symptoms']}<br>
    • <strong>Symptom Breakdown</strong>: {stats['symptom_breakdown']['observations']} observations, {stats['symptom_breakdown']['conditions']} conditions<br>
    • <strong>Network Connectivity</strong>: {results['network_statistics']['nodes']} symptoms, {results['network_statistics']['edges']} correlations<br>
    • <strong>Average Clustering</strong>: {results['network_statistics']['avg_clustering']:.3f}
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive Network Visualization
    if results['network_statistics']['edges'] > 0:
        st.subheader("🕸️ Interactive Symptom Correlation Network")
        st.markdown("**Hover over nodes to see symptom details. Click and drag to explore!**")
        
        with st.spinner("Generating interactive network visualization..."):
            try:
                net = analyzer._create_interactive_pyvis_network(analyzer.G)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                    net.save_graph(tmp_file.name)
                    
                    # Read and display the HTML
                    with open(tmp_file.name, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    components.html(html_content, height=750)
                    
                    # Clean up
                    os.unlink(tmp_file.name)
            except Exception as e:
                st.error(f"Error creating network visualization: {str(e)}")
        
        # Legend
        st.markdown("""
        **Legend:**
        - **Node Color**: Blue (few connections) → Red (many connections)
        - **Node Size**: Based on number of correlations
        - **Edge Color**: Blue (positive correlation), Red (negative correlation)
        - **Edge Thickness**: Strength of correlation
        """)


    else:
        st.warning("No correlations above threshold. Try lowering the minimum correlation.")
    
    # Top Patterns
    with st.expander("🔍 Top Co-occurring Patterns", expanded=True):
        if results['co_occurring_patterns']['total_patterns'] > 0:
            patterns = results['co_occurring_patterns']['patterns'][:20]
            
            pattern_data = []
            for i, pattern in enumerate(patterns):
                # Clean symptom names for display
                clean_symptoms = []
                for s in pattern['symptoms']:
                    clean_s = s.split('|')[-1].strip() if '|' in s else s
                    if len(clean_s) > 40:
                        clean_s = clean_s[:37] + '...'
                    clean_symptoms.append(clean_s)
                
                pattern_data.append({
                    'Rank': i + 1,
                    'Pattern': ' + '.join(clean_symptoms),
                    'Size': pattern['pattern_size'],
                    'Patients': pattern['affected_patients'],
                    'Prevalence': f"{pattern['pattern_prevalence']*100:.1f}%"
                })
            
            pattern_df = pd.DataFrame(pattern_data)
            st.dataframe(pattern_df, use_container_width=True, hide_index=True)
            
            # Visualization
            if len(patterns) > 0:
                top_10 = patterns[:10]
                
                # Clean labels for chart
                clean_labels = []
                for p in top_10:
                    clean_symp = [s.split('|')[-1].strip() if '|' in s else s for s in p['symptoms']]
                    clean_symp = [s[:20] + '...' if len(s) > 20 else s for s in clean_symp]
                    clean_labels.append(' + '.join(clean_symp))
                
                fig = px.bar(
                    x=[p['affected_patients'] for p in top_10],
                    y=clean_labels,
                    orientation='h',
                    title="Top 10 Co-occurring Patterns by Patient Count",
                    labels={'x': 'Number of Patients', 'y': 'Pattern'},
                    color=[p['pattern_prevalence'] for p in top_10],
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No co-occurring patterns found with current parameters.")
    
    # Top Correlations
    with st.expander("📈 Top Symptom Correlations"):
        if results['symptom_correlations']['total_pairs'] > 0:
            correlations = results['symptom_correlations']['top_correlations'][:30]
            
            corr_data = []
            for i, corr in enumerate(correlations):
                # Clean names for display
                s1 = corr['symptom_1'].split('|')[-1].strip() if '|' in corr['symptom_1'] else corr['symptom_1']
                s2 = corr['symptom_2'].split('|')[-1].strip() if '|' in corr['symptom_2'] else corr['symptom_2']
                
                s1 = s1[:40] + '...' if len(s1) > 40 else s1
                s2 = s2[:40] + '...' if len(s2) > 40 else s2
                
                corr_data.append({
                    'Rank': i + 1,
                    'Symptom 1': s1,
                    'Symptom 2': s2,
                    'Correlation': f"{corr['correlation']:.3f}",
                    'Co-occurrence': corr['co_occurrence_count']
                })
            
            corr_df = pd.DataFrame(corr_data)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
            
            # Heatmap of top correlations
            if len(correlations) >= 10:
                top_symptoms = set()
                for corr in correlations[:15]:
                    top_symptoms.add(corr['symptom_1'])
                    top_symptoms.add(corr['symptom_2'])
                top_symptoms = list(top_symptoms)[:10]
                
                # Create correlation matrix
                corr_matrix = pd.DataFrame(0.0, index=top_symptoms, columns=top_symptoms)
                for corr in correlations:
                    if corr['symptom_1'] in top_symptoms and corr['symptom_2'] in top_symptoms:
                        corr_matrix.loc[corr['symptom_1'], corr['symptom_2']] = corr['correlation']
                        corr_matrix.loc[corr['symptom_2'], corr['symptom_1']] = corr['correlation']
                
                # Clean labels for heatmap
                clean_labels = [s.split('|')[-1].strip() if '|' in s else s for s in top_symptoms]
                clean_labels = [s[:30] + '...' if len(s) > 30 else s for s in clean_labels]
                corr_matrix.index = clean_labels
                corr_matrix.columns = clean_labels
                
                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Heatmap (Top Symptoms)",
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No correlations found.")
    
    # Symptom Prevalence
    with st.expander("📊 Symptom Prevalence Distribution"):
        prevalence_data = results['symptom_prevalence'][:30]
        
        if prevalence_data:
            prev_df = pd.DataFrame(prevalence_data)
            # Clean symptom names
            prev_df['symptom_short'] = prev_df['symptom'].apply(
                lambda x: (x.split('|')[-1].strip() if '|' in x else x)[:30] + '...' 
                if len(x.split('|')[-1].strip() if '|' in x else x) > 30 
                else (x.split('|')[-1].strip() if '|' in x else x)
            )
            
            fig = px.bar(
                prev_df,
                x='prevalence',
                y='symptom_short',
                orientation='h',
                title="Top 30 Symptoms by Prevalence",
                labels={'prevalence': 'Prevalence', 'symptom_short': 'Symptom'},
                color='prevalence',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=700, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Associations
    if results['statistical_associations']['total_significant'] > 0:
        with st.expander("🔬 Statistical Associations (Chi-Square Test)"):
            associations = results['statistical_associations']['associations'][:20]
            
            assoc_data = []
            for i, assoc in enumerate(associations):
                # Clean names
                s1 = assoc['symptom_1'].split('|')[-1].strip() if '|' in assoc['symptom_1'] else assoc['symptom_1']
                s2 = assoc['symptom_2'].split('|')[-1].strip() if '|' in assoc['symptom_2'] else assoc['symptom_2']
                
                s1 = s1[:40] + '...' if len(s1) > 40 else s1
                s2 = s2[:40] + '...' if len(s2) > 40 else s2
                
                assoc_data.append({
                    'Rank': i + 1,
                    'Symptom 1': s1,
                    'Symptom 2': s2,
                    'Chi-Square': f"{assoc['chi_square']:.2f}",
                    'P-Value': f"{assoc['p_value']:.4f}",
                    'Significant': '✓' if assoc['significant'] else '✗'
                })
            
            assoc_df = pd.DataFrame(assoc_data)
            st.dataframe(assoc_df, use_container_width=True, hide_index=True)
