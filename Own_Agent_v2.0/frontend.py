import streamlit as st
import requests
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

API_BASE = "http://localhost:8000"

# Enhanced page config
st.set_page_config(
    page_title="My Own Agent", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        min-height: 100vh;
        padding: 0;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 3rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .main-title { 
        font-size: 3.5rem; 
        font-weight: 700; 
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        font-size: 1.25rem;
        font-weight: 300;
        opacity: 0.85;
        margin: 0;
    }
    
    .step-card {
        background: white;
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border-left: 4px solid #3b82f6;
    }
    
    .step-title { 
        font-size: 2rem; 
        font-weight: 600; 
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    }
    
    .step-number {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        width: 45px;
        height: 45px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        font-weight: 700;
        font-size: 1.25rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        padding: 2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0;
        box-shadow: 0 1px 3px rgba(59,130,246,0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .status-running {
        background: #fef3cd;
        color: #856404;
        border: 1px solid #f0e68c;
    }
    
    .status-completed {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-failed {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f1b0b7;
    }
    
    .pipeline-history-item {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border-left: 3px solid #3b82f6;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .pipeline-history-item:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .file-upload-zone {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 4rem 2rem;
        text-align: center;
        background: #f8fafc;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .file-upload-zone:hover {
        background: #f1f5f9;
        border-color: #3b82f6;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: left;
        transition: transform 0.2s ease;
        border: 1px solid #e2e8f0;
        height: 100%;
        margin-bottom: 1rem;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .feature-card h3 {
        color: #1e293b;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .feature-card p {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0;
    }
    
    .alert-info {
        background: #dbeafe;
        color: #1e40af;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bfdbfe;
        margin: 1rem 0;
    }
    
    .alert-success {
        background: #dcfce7;
        color: #166534;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bbf7d0;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: #fef3c7;
        color: #92400e;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #fde68a;
        margin: 1rem 0;
    }
    
    .sidebar .element-container {
        background: white;
        margin: 0.5rem 0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f8fafc;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        border: 1px solid #e2e8f0;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: #3b82f6;
        color: white;
        border-color: #3b82f6;
    }
    
    .progress-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .tab-content {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    
    .config-summary {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-color: #3b82f6;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit components */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Spacing adjustments */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* Improved form styling */
    .stSelectbox label, .stTextInput label, .stRadio label {
        font-weight: 500;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    /* Table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Navigation styling */
    .nav-buttons {
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state with advanced mode and notifications default
def initialize_session():
    defaults = {
        "step": 1,
        "history": [],
        "advanced_mode": True,
        "notifications": True,
        "auto_refresh": True,
        "pipeline_configs": {},
        "recent_uploads": [],
        "favorites": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session()

# Header
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">My Own Agent</div>
        <div class="subtitle">Machine Learning Pipeline Automation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Enhanced Helper Functions
def save_uploaded_file(uploaded_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}_{uploaded_file.name}"
    save_path = os.path.join("data/csv_data", file_name)
    os.makedirs("data/csv_data", exist_ok=True)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if "recent_uploads" not in st.session_state:
        st.session_state["recent_uploads"] = []
    
    upload_info = {
        "filename": file_name,
        "original_name": uploaded_file.name,
        "timestamp": timestamp,
        "size": uploaded_file.size
    }
    st.session_state["recent_uploads"].insert(0, upload_info)
    st.session_state["recent_uploads"] = st.session_state["recent_uploads"][:10]
    
    return file_name

def get_dataset_preview(file_path):
    try:
        full_path = os.path.join("data/csv_data", file_path)
        df = pd.read_csv(full_path, nrows=100)
        return df
    except Exception as e:
        st.error(f"Error reading dataset: {str(e)}")
        return None

def show_enhanced_results(folder, report_name, section_name):
    if not os.path.exists(folder):
        st.markdown(f"""
        <div class="alert-info">
            <strong>No {section_name} results found yet.</strong><br>
            Run the corresponding pipeline stage to generate results.
        </div>
        """, unsafe_allow_html=True)
        return

    report_key = f"{folder}_{report_name}"
    if report_key not in st.session_state:
        report_path = os.path.join(folder, report_name)
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                st.session_state[report_key] = f.read()

    col1, col2 = st.columns([3, 1])
    
    with col1:
        if report_key in st.session_state:
            with st.expander(f"{section_name} Report", expanded=True):
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                st.markdown(st.session_state[report_key])
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.download_button(
                    label=f"Download {section_name} Report",
                    data=st.session_state[report_key],
                    file_name=f"{section_name.lower()}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )

    with col2:
        st.markdown("### Quick Stats")
        
        if os.path.exists(folder):
            files = os.listdir(folder)
            st.metric("Generated Files", len(files))
            
            folder_mtime = os.path.getmtime(folder)
            last_modified = datetime.fromtimestamp(folder_mtime).strftime("%Y-%m-%d %H:%M")
            st.metric("Last Updated", last_modified)

    img_files = []
    if os.path.exists(folder):
        img_files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
    
    if img_files:
        st.markdown(f"### {section_name} Visualizations")
        
        cols = st.columns(min(3, len(img_files)))
        for i, img in enumerate(img_files):
            with cols[i % len(cols)]:
                img_path = os.path.join(folder, img)
                st.image(img_path, caption=img.replace('_', ' ').title(), use_container_width=True)
                
                with open(img_path, "rb") as f:
                    st.download_button(
                        label=f"Download {img}",
                        data=f.read(),
                        file_name=img,
                        mime="image/png",
                        key=f"download_{img}_{i}"
                    )

    if os.path.exists(folder):
        model_files = [f for f in os.listdir(folder) if f.endswith((".pkl", ".joblib", ".h5", ".pt"))]
        if model_files:
            st.markdown("### Model Files")
            for model_file in model_files:
                model_path = os.path.join(folder, model_file)
                file_size = os.path.getsize(model_path)
                file_size_mb = file_size / (1024 * 1024)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{model_file}**")
                with col2:
                    st.write(f"{file_size_mb:.2f} MB")
                with col3:
                    with open(model_path, "rb") as f:
                        st.download_button(
                            label="Download",
                            data=f.read(),
                            file_name=model_file,
                            mime="application/octet-stream",
                            key=f"model_download_{model_file}"
                        )

# Enhanced Sidebar
with st.sidebar:
    st.markdown("## Pipeline History")
    
    if st.session_state["history"]:
        for i, pipeline_id in enumerate(st.session_state["history"][:5]):
            pipeline_info = st.session_state.get("pipeline_configs", {}).get(pipeline_id, {})
            created_time = pipeline_info.get("created_time", "Unknown")
            dataset_name = pipeline_info.get("dataset_name", "Unknown Dataset")
            
            if st.button(f"{pipeline_id[:8]}...\n{dataset_name}\n{created_time}", key=f"hist_{i}"):
                st.session_state["pipeline_id"] = pipeline_id
                st.session_state["dataset_name"] = dataset_name
                st.session_state["step"] = 4
                st.rerun()
    else:
        st.markdown("""
        <div class="alert-info">
            <strong>No pipeline history yet.</strong><br>
            Run your first pipeline to see it here.
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state["history"]:
        st.markdown("### Statistics")
        st.metric("Total Pipelines", len(st.session_state["history"]))
        st.metric("Recent Uploads", len(st.session_state.get("recent_uploads", [])))

# WIZARD STEPS
step = st.session_state.get("step", 1)

# Step 1: Upload Dataset
if step == 1:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">
            <div class="step-number">1</div>
            Upload Your Dataset
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="file-upload-zone">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose your CSV file", 
            type=["csv"],
            help="Upload a CSV file containing your dataset. Maximum size: 200MB"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
        if uploaded_file:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f'Uploading... {i+1}%')
                time.sleep(0.01)
            
            file_path = save_uploaded_file(uploaded_file)
            st.session_state["file_path"] = file_path
            
            st.markdown("""
            <div class="alert-success">
                <strong>File uploaded successfully!</strong><br>
                Your dataset has been processed and is ready for analysis.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Dataset Preview")
            df = get_dataset_preview(file_path)
            if df is not None:
                st.dataframe(df, use_container_width=True, height=300)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Rows</div>
                    </div>
                    """.format(len(df)), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Columns</div>
                    </div>
                    """.format(len(df.columns)), unsafe_allow_html=True)
                
                with col3:
                    numeric_cols = df.select_dtypes(include=['number']).shape[1]
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Numeric</div>
                    </div>
                    """.format(numeric_cols), unsafe_allow_html=True)
                
                with col4:
                    categorical_cols = df.select_dtypes(include=['object']).shape[1]
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Categorical</div>
                    </div>
                    """.format(categorical_cols), unsafe_allow_html=True)
            
            st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
            if st.button("Next: Configure Pipeline", type="primary", use_container_width=True):
                st.session_state["step"] = 2
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Recent Uploads")
        recent_uploads = st.session_state.get("recent_uploads", [])
        
        if recent_uploads:
            for upload in recent_uploads[:5]:
                if st.button(f"{upload['original_name'][:20]}...\n{upload['timestamp']}", key=f"recent_{upload['timestamp']}"):
                    st.session_state["file_path"] = upload['filename']
                    st.session_state["step"] = 2
                    st.rerun()
        else:
            st.info("No recent uploads")

# Step 2: Configuration
elif step == 2:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">
            <div class="step-number">2</div>
            Configure Your Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    df = None
    if "file_path" in st.session_state:
        df = get_dataset_preview(st.session_state["file_path"])
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Machine Learning Objective")
        
        objective_options = ["classification", "regression"]
        
        selected_objective = st.radio(
            "Choose your ML objective:",
            objective_options,
            horizontal=True
        )
        st.session_state["objective"] = selected_objective
        
        st.markdown("### Target Configuration")
        
        if df is not None:
            suggested_targets = []
            for col in df.columns:
                if 'target' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
                    suggested_targets.append(col)
            
            target_column = st.selectbox(
                "Select target column:",
                options=df.columns.tolist(),
                index=df.columns.tolist().index(suggested_targets[0]) if suggested_targets else 0,
                help="This is the column your model will learn to predict"
            )
        else:
            target_column = st.text_input("Target Column", value="label", help="Enter the name of your target column")
        
        st.session_state["target_column"] = target_column
        
        # Advanced configuration (always shown since advanced_mode is default)
        st.markdown("### Advanced Settings")
        
        with st.expander("Model Configuration", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            with col_b:
                random_state = st.number_input("Random State", 0, 9999, 42)
                max_time = st.slider("Max Training Time (minutes)", 5, 120, 30)
            
            st.session_state["ml_config"] = {
                "test_size": test_size,
                "cv_folds": cv_folds,
                "random_state": random_state,
                "max_time_mins": max_time
            }
        
        with st.expander("Preprocessing Options", expanded=False):
            handle_missing = st.selectbox("Handle Missing Values", ["auto", "drop", "impute"])
            scale_features = st.toggle("Feature Scaling", True)
            feature_selection = st.toggle("Automatic Feature Selection", False)
            
            st.session_state["preprocessing_config"] = {
                "handle_missing": handle_missing,
                "scale_features": scale_features,
                "feature_selection": feature_selection
            }
    
    with col2:
        st.markdown("### Configuration Summary")
        
        config_summary = f"""
        <div class="config-summary">
        <strong>Objective:</strong> {st.session_state.get('objective', 'Not set').title()}<br><br>
        <strong>Target Column:</strong> {st.session_state.get('target_column', 'Not set')}<br><br>
        """
        
        config_summary += "</div>"
        st.markdown(config_summary, unsafe_allow_html=True)
        
        config_valid = all([
            st.session_state.get("objective"),
            st.session_state.get("target_column")
        ])
    
    st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Back to Upload", use_container_width=True):
            st.session_state["step"] = 1
            st.rerun()
    with col3:
        if config_valid and st.button("Next: Execute Pipeline", type="primary", use_container_width=True):
            st.session_state["step"] = 3
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Step 3: Execution
elif step == 3:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">
            <div class="step-number">3</div>
            Execute Your Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Pipeline Execution Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>Full AutoML Pipeline</h3>
            <p>End-to-end machine learning pipeline including EDA, preprocessing, training, optimization, and evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run Full Pipeline", type="primary", use_container_width=True):
            payload = {
                "file_path": st.session_state["file_path"],
                "objective": st.session_state["objective"],
                "target_column": st.session_state["target_column"],
            }
            
            if "ml_config" in st.session_state:
                payload.update(st.session_state["ml_config"])
            
            with st.spinner("Launching full pipeline..."):
                try:
                    response = requests.post(f"{API_BASE}/run/automl/full", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state["pipeline_id"] = result["pipeline_id"]
                        st.session_state["dataset_name"] = st.session_state["file_path"].replace(".csv", "")
                        
                        pipeline_config = {
                            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "dataset_name": st.session_state["dataset_name"],
                            "objective": st.session_state["objective"],
                            "target_column": st.session_state["target_column"],
                            "type": "full_pipeline"
                        }
                        st.session_state["pipeline_configs"][result["pipeline_id"]] = pipeline_config
                        
                        st.session_state["history"].insert(0, result["pipeline_id"])
                        st.session_state["step"] = 4
                        
                        st.success(f"Full pipeline launched successfully!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"Error launching pipeline: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
        
        st.markdown("""
        <div class="feature-card">
            <h3>Exploratory Data Analysis</h3>
            <p>Comprehensive data analysis with statistics, visualizations, and insights about the dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run EDA Only", use_container_width=True):
            payload = {
                "file_path": st.session_state["file_path"], 
                "objective": st.session_state["objective"]
            }
            
            with st.spinner("Running exploratory data analysis..."):
                try:
                    response = requests.post(f"{API_BASE}/run/automl/eda", json=payload)
                    if response.status_code == 200:
                        st.session_state["dataset_name"] = st.session_state["file_path"].replace(".csv", "")
                        st.session_state["step"] = 4
                        st.success("EDA completed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error running EDA: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>Data Preprocessing</h3>
            <p>Clean and prepare the dataset with missing value handling, feature encoding, and scaling.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run Preprocessing", use_container_width=True):
            payload = {
                "file_path": st.session_state["file_path"],
                "objective": st.session_state["objective"],
                "target_column": st.session_state["target_column"],
            }
            
            if "preprocessing_config" in st.session_state:
                payload["preprocessing_config"] = st.session_state["preprocessing_config"]
            
            with st.spinner("Processing your data..."):
                try:
                    response = requests.post(f"{API_BASE}/run/automl/preprocessing", json=payload)
                    if response.status_code == 200:
                        st.session_state["dataset_name"] = st.session_state["file_path"].replace(".csv", "")
                        st.session_state["step"] = 4
                        st.success("Preprocessing completed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error in preprocessing: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
        
        st.markdown("""
        <div class="feature-card">
            <h3>Model Training</h3>
            <p>Train machine learning models and compare their performance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run Training Only", use_container_width=True):
            payload = {
                "file_path": st.session_state["file_path"],
                "objective": st.session_state["objective"],
                "ml_model_config": st.session_state.get("ml_config", {}),
            }
            
            with st.spinner("Training models..."):
                try:
                    response = requests.post(f"{API_BASE}/run/automl/training", json=payload)
                    if response.status_code == 200:
                        st.session_state["dataset_name"] = st.session_state["file_path"].replace(".csv", "")
                        st.session_state["step"] = 4
                        st.success("Model training completed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error in training: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
    
    st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Back to Configuration", use_container_width=True):
            st.session_state["step"] = 2
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Step 4: Results and Monitoring
elif step == 4:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">
            <div class="step-number">4</div>
            Monitor & Analyze Results
        </div>
    </div>
    """, unsafe_allow_html=True)

    dataset_name = st.session_state.get("dataset_name", "unknown")
    base_dir = "outputs/AutoML_Output"

    # Live Monitoring
    if "pipeline_id" in st.session_state:
        pipeline_id = st.session_state["pipeline_id"]
        
        st.markdown("### Live Pipeline Monitoring")
        
        monitoring_container = st.container()
        
        with monitoring_container:
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            logs_placeholder = st.empty()
            
            auto_refresh = st.session_state.get("auto_refresh", True)
            
            if auto_refresh:
                refresh_interval = st.slider("Refresh Interval (seconds)", 1, 30, 5)
            else:
                if st.button("Refresh Status"):
                    pass
            
            max_iterations = 60 if auto_refresh else 1
            for i in range(max_iterations):
                try:
                    status_response = requests.get(f"{API_BASE}/status/{pipeline_id}")
                    if status_response.status_code == 200:
                        status = status_response.json()
                        
                        with progress_placeholder.container():
                            progress_value = int(status.get("progress", 0))
                            st.progress(progress_value, text=f"Pipeline Progress: {progress_value}%")
                            
                            current_status = status.get("status", "unknown")
                            if current_status == "running":
                                st.markdown('<div class="status-badge status-running">Running</div>', unsafe_allow_html=True)
                            elif current_status == "completed":
                                st.markdown('<div class="status-badge status-completed">Completed</div>', unsafe_allow_html=True)
                            elif current_status == "failed":
                                st.markdown('<div class="status-badge status-failed">Failed</div>', unsafe_allow_html=True)
                            
                            if "current_stage" in status:
                                st.info(f"Current Stage: {status['current_stage']}")
                            
                            if "eta_minutes" in status:
                                st.metric("Estimated Time Remaining", f"{status['eta_minutes']} minutes")
                        
                        with status_placeholder.container():
                            with st.expander("Detailed Status", expanded=False):
                                st.json(status)
                        
                        if "logs" in status and status["logs"]:
                            with logs_placeholder.container():
                                st.markdown("### Live Logs")
                                log_container = st.container()
                                with log_container:
                                    for log_entry in status["logs"][-10:]:
                                        st.text(log_entry)
                        
                        if status["status"] in ["completed", "failed"]:
                            if status["status"] == "completed":
                                st.success("Pipeline completed successfully!")
                                st.balloons()
                            elif status["status"] == "failed":
                                st.error(f"Pipeline failed: {status.get('error', 'Unknown error')}")
                            break
                    else:
                        st.warning(f"Unable to fetch status (HTTP {status_response.status_code})")
                        break
                
                except Exception as e:
                    st.error(f"Error fetching status: {str(e)}")
                    break
                
                if auto_refresh and i < max_iterations - 1:
                    time.sleep(refresh_interval)
                else:
                    break

    # Results Display with Tabs
    st.markdown("### Pipeline Results")
    
    tab_eda, tab_pre, tab_train, tab_opt, tab_eval, tab_compare = st.tabs([
        "EDA", "Preprocessing", "Training", "Optimization", "Evaluation", "Compare"
    ])

    with tab_eda:
        show_enhanced_results(
            f"{base_dir}/{dataset_name}_EDA", 
            "eda_report.md", 
            "Exploratory Data Analysis"
        )

    with tab_pre:
        show_enhanced_results(
            f"{base_dir}/{dataset_name}_preprocessing", 
            "preprocessing_report.md", 
            "Data Preprocessing"
        )

    with tab_train:
        show_enhanced_results(
            f"{base_dir}/{dataset_name}_training", 
            "model_training_report.md", 
            "Model Training"
        )

    with tab_opt:
        show_enhanced_results(
            f"{base_dir}/{dataset_name}_optimization", 
            "optimization_report.md", 
            "Hyperparameter Optimization"
        )

    with tab_eval:
        show_enhanced_results(
            f"{base_dir}/{dataset_name}_final_evaluation", 
            "final_evaluation_report.md", 
            "Final Model Evaluation"
        )
    
    with tab_compare:
        st.markdown("### Model Comparison")
        
        comparison_data = []
        
        comparison_file = f"{base_dir}/{dataset_name}_final_evaluation/model_comparison.json"
        if os.path.exists(comparison_file):
            try:
                with open(comparison_file, 'r') as f:
                    comparison_data = json.load(f)
                
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    st.dataframe(df_comparison, use_container_width=True)
                    
                    if len(df_comparison) > 1:
                        fig = px.bar(
                            df_comparison, 
                            x='model_name', 
                            y='accuracy' if 'accuracy' in df_comparison.columns else 'score',
                            title="Model Performance Comparison",
                            color='model_name'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No model comparison data available yet.")
            except Exception as e:
                st.error(f"Error loading comparison data: {str(e)}")
        else:
            st.info("Run the full pipeline to generate model comparisons.")

    # Action Center
    st.markdown("### Action Center")
    
    st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("Run New Pipeline", type="primary", use_container_width=True):
            keys_to_keep = ["history", "advanced_mode", "notifications", "auto_refresh", "pipeline_configs"]
            keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]
            for key in keys_to_remove:
                del st.session_state[key]
            st.session_state["step"] = 1
            st.rerun()
    
    with action_col2:
        if st.button("Back to Execute", use_container_width=True):
            st.session_state["step"] = 3
            st.rerun()
    
    with action_col3:
        if st.button("Export All Results", use_container_width=True):
            st.info("Export functionality will be implemented soon.")
    
    with action_col4:
        if "pipeline_id" in st.session_state:
            if st.button("Add to Favorites", use_container_width=True):
                if "favorites" not in st.session_state:
                    st.session_state["favorites"] = []
                
                if st.session_state["pipeline_id"] not in st.session_state["favorites"]:
                    st.session_state["favorites"].append(st.session_state["pipeline_id"])
                    st.success("Added to favorites!")
                else:
                    st.info("Already in favorites!")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.8); border-radius: 8px; margin-top: 2rem; color: #64748b;'>
    <p>Built with Streamlit | GMI </p>
</div>
""", unsafe_allow_html=True)