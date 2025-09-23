import streamlit as st
import requests
import json
import pandas as pd
import os
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="My Own Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Constants
API_BASE_URL = "http://localhost:8000"

def check_api_status():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_tools():
    """Get list of available MCP tools"""
    try:
        response = requests.get(f"{API_BASE_URL}/tools", timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def run_automl_pipeline(dataset_path, target_column, task_type, output_path, dataset_name):
    """Run the AutoML pipeline"""
    try:
        payload = {
            "dataset_path": dataset_path,
            "target_column": target_column,
            "task_type": task_type,
            "output_path": output_path,
            "dataset_name": dataset_name
        }
        
        response = requests.post(
            f"{API_BASE_URL}/run_automl", 
            json=payload,
            timeout=300  # 5 minutes timeout
        )
        
        return response.json(), response.status_code
    except requests.exceptions.Timeout:
        return {"status": "error", "error": "Request timed out. The process might still be running."}, 408
    except Exception as e:
        return {"status": "error", "error": str(e)}, 500

def main():
    # Header
    st.markdown('<div class="main-header">My Own Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AutoML with CrewAI and MCP Tools</div>', unsafe_allow_html=True)
    
    # Check API status
    api_status = check_api_status()
    
    # Sidebar
    with st.sidebar:
        st.header("Status")
        
        # API Status
        if api_status:
            st.markdown('<div class="success-box">API Server: Online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">API Server: Offline</div>', unsafe_allow_html=True)
            st.error("Make sure to run: `uvicorn main:app --reload`")
            st.stop()
        
        st.markdown("---")
        
        # Tools information
        if st.button("Check Available Tools"):
            with st.spinner("Checking available tools..."):
                tools_info = get_available_tools()
                if tools_info and tools_info.get("status") != "error":
                    st.success(f"Found {tools_info.get('count', 0)} tools")
                    with st.expander("View Tools"):
                        for tool in tools_info.get('tools', []):
                            st.write(f"**{tool['name']}**")
                            if tool.get('description'):
                                st.write(tool['description'])
                            st.write("---")
                else:
                    st.error("Could not retrieve tools. Make sure MCP server is running.")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Dataset Configuration")
        
        # Dataset upload or path input
        upload_option = st.radio(
            "How would you like to provide your dataset?",
            ["Upload File", "Specify File Path"],
            horizontal=True
        )
        
        dataset_path = None
        dataset_name = None
        
        if upload_option == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload your dataset", 
                type=['csv', 'xlsx', 'json', 'parquet'],
                help="Supported formats: CSV, Excel, JSON, Parquet"
            )
            
            if uploaded_file:
                # Save uploaded file
                upload_dir = Path("uploads")
                upload_dir.mkdir(exist_ok=True)
                dataset_path = upload_dir / uploaded_file.name
                
                with open(dataset_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                dataset_name = uploaded_file.name.split('.')[0]
                st.success(f"File uploaded: {uploaded_file.name}")
                
                # Preview dataset
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(dataset_path)
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(dataset_path)
                    elif uploaded_file.name.endswith('.json'):
                        df = pd.read_json(dataset_path)
                    elif uploaded_file.name.endswith('.parquet'):
                        df = pd.read_parquet(dataset_path)
                    
                    st.subheader("Dataset Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        else:
            dataset_path = st.text_input(
                "Dataset File Path",
                placeholder="/path/to/your/dataset.csv",
                help="Provide the full path to your dataset file"
            )
            
            dataset_name = st.text_input(
                "Dataset Name (optional)",
                placeholder="my_dataset",
                help="A friendly name for your dataset"
            )
            
            # If no dataset_name provided, derive it from the path
            if dataset_path and not dataset_name:
                dataset_name = Path(dataset_path).stem

        # Output path configuration
        output_path = st.text_input(
            "Output Directory",
            value="./output",
            help="Directory where results will be saved"
        )
        
        # ML Configuration
        st.header("ML Configuration")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            target_column = st.text_input(
                "Target Column",
                placeholder="target",
                help="The column you want to predict"
            )
        
        with col_b:
            task_type = st.selectbox(
                "Task Type",
                ["classification", "regression"],
                help="Type of machine learning task"
            )

    
    with col2:
        st.header("Run Pipeline")
        
        # Validation
        can_run = bool(dataset_path and target_column and api_status)
        
        if not can_run:
            missing_items = []
            if not dataset_path:
                missing_items.append("Dataset path")
            if not target_column:
                missing_items.append("Target column")
            if not api_status:
                missing_items.append("API connection")
            
            st.markdown(f'<div class="info-box">Missing: {", ".join(missing_items)}</div>', unsafe_allow_html=True)
        
        # Run button
        if st.button("Start AutoML Agent", disabled=not can_run, use_container_width=True):
            with st.spinner("Running AutoML pipeline... This may take several minutes."):
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress updates
                for i in range(10):
                    progress_bar.progress((i + 1) * 10)
                    status_text.text(f"Processing... ({(i + 1) * 10}%)")
                    time.sleep(0.5)
                
                # Run the actual pipeline - FIXED: Now passing all 5 required parameters
                result, status_code = run_automl_pipeline(
                    str(dataset_path),
                    target_column,
                    task_type,
                    output_path,  # Added this missing parameter
                    dataset_name or "dataset"  # Moved this to the correct position
                )
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                
                # Display results
                if status_code == 200 and result.get("status") == "success":
                    st.markdown('<div class="success-box">AutoML pipeline completed successfully!</div>', unsafe_allow_html=True)
                    
                    # Show results
                    st.subheader("Results")
                    
                    with st.expander("View Full Results"):
                        st.json(result)
                    
                    # Show request parameters
                    st.subheader("Parameters Used")
                    params = result.get("request_parameters", {})
                    for key, value in params.items():
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                else:
                    st.markdown('<div class="error-box">Pipeline failed</div>', unsafe_allow_html=True)
                    st.error(result.get("error", "Unknown error occurred"))
                    
                    if result.get("suggestion"):
                        st.info(result["suggestion"])
                    
                    with st.expander("View Error Details"):
                        st.json(result)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            Built with using Streamlit, CrewAI, and MCP Tools
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()