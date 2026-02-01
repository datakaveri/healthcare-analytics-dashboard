# 🏥 Medical Analytics Dashboard

A professional, minimalistic Streamlit application for comprehensive medical data analysis and visualization.

## ✨ Features

### 📊 **Data Overview**
- Interactive data tables with pagination
- Comprehensive data summary and statistics
- Missing value analysis

### 🔗 **Correlation Analysis**
- Interactive correlation heatmaps
- Cross-correlation between observations and conditions
- Top correlation identification

### 🎯 **Patient Clustering**
- K-means clustering with PCA visualization
- Interactive scatter plots with cluster centers
- Detailed cluster analysis and patient assignments

### 🚨 **Anomaly Detection**
- Isolation Forest algorithm implementation
- Anomalous patient identification
- Symptom pattern comparison between normal and anomalous cases

### 🧬 **Patient Phenotyping**
- Optimal cluster identification using silhouette scores
- Interactive radar charts for phenotype visualization
- Comprehensive phenotype comparison heatmaps

### 📈 **Statistical Analysis**
- Descriptive statistics (means, medians, modes)
- Standard deviation and variance analysis
- Range analysis and coefficient of variation
- Correlation and covariance calculations

### 📊 **Prevalence Analysis**
- Condition prevalence calculations
- Interactive prevalence visualizations
- Patient segmentation by demographic variables

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Internet access to reach the FHIR server

### Installation

1. **Clone or download the project**
   ```bash
   cd streamlit-demo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
    python run.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8050?databankId=LeptoDemo` (replace `LeptoDemo` with your group id)

## 🔗 Live FHIR Integration

The app now loads patients and observations/conditions directly from a FHIR server.

- **FHIR Base URL**: `https://fhir.rs.adarv.in/`
- **Group Id (databankId)**: Read from the page URL query parameters. Example:
  - `http://localhost:8050?databankId=LeptoDemo`
  - You can also enter or override the group id from the sidebar input.

The app builds `patients_df` and the processed dataset on the fly for the specified group and feeds them into all analysis modules.

## 📁 Project Structure

```
streamlit-demo/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── modules/                       # Analysis modules
│   ├── correlation_analysis.py   # Correlation analysis functions
│   ├── clustering_analysis.py    # Patient clustering functions
│   ├── anomaly_detection.py      # Anomaly detection functions
│   ├── phenotyping_analysis.py   # Patient phenotyping functions
│   ├── statistical_analysis.py   # Statistical analysis functions
│   └── prevalence_analysis.py    # Prevalence analysis functions
├── run.py                        # Runner that starts Streamlit on port 8050
```

## 🎨 Design Features

- **Professional Minimalistic Design**: Clean, modern interface with subtle gradients and professional color schemes
- **Responsive Layout**: Optimized for different screen sizes
- **Interactive Visualizations**: Plotly-powered charts with hover effects and zoom capabilities
- **Smooth Navigation**: Intuitive sidebar navigation with clear section organization
- **Performance Optimized**: Cached data loading and efficient processing

## 📊 Data Requirements

The application expects the following data files:

- **`processed_data.csv`**: Main dataset with patient medical information
- **`patients_df.csv`**: Patient demographic and basic information
- **`obs_names.pkl`** (optional): List of observation names for better labeling
- **`cond_names.pkl`** (optional): List of condition names for better labeling

## 🔧 Customization

### Adding New Analysis Modules

1. Create a new Python file in the `modules/` directory
2. Implement your analysis function following the existing pattern
3. Import and add the function to `app.py`
4. Add a new navigation option in the sidebar

### Styling Customization

The application uses custom CSS for styling. You can modify the styles in the `st.markdown()` section at the beginning of `app.py`.

## 🐛 Troubleshooting

### Common Issues

1. **Module Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **Data File Not Found**: Verify that `processed_data.csv` and `patients_df.csv` are in the project root directory
3. **Performance Issues**: For large datasets, consider implementing data sampling or caching strategies

### Performance Tips

- The application uses `@st.cache_data` for data loading optimization
- Large datasets may require pagination or sampling for better performance
- Consider using `st.empty()` for dynamic content updates

## 📈 Future Enhancements

- [ ] Real-time data integration
- [ ] Export functionality for reports and visualizations
- [ ] Advanced filtering and search capabilities
- [ ] Machine learning model integration
- [ ] Multi-user authentication and role-based access
- [ ] API integration for external data sources

---
