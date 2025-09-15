import streamlit as st
import requests
import json
import time
import os
from typing import Dict, Any
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for modern styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    /* Task Cards */
    .task-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .task-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        border-color: #667eea;
    }
    
    .task-card-selected {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-color: #f5576c;
    }
    
    .task-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .task-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .task-description {
        font-size: 0.95rem;
        opacity: 0.8;
        line-height: 1.5;
    }
    
    /* Status Cards */
    .status-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
    }
    
    .status-card.success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .status-card.error {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
    }
    
    .status-card.warning {
        background: linear-gradient(135deg, #fdbb2d 0%, #22c1c3 100%);
    }
    
    /* Forms */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e1e5e9;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f7f8fc 0%, #ffffff 100%);
    }
    
    /* Results Container */
    .results-container {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #f0f0f0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 3rem;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* File Browser */
    .file-item {
        background: white;
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .file-item:hover {
        background: #f9fafb;
        border-color: #667eea;
    }
    
    .file-icon {
        margin-right: 0.5rem;
        font-size: 1.2rem;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    </style>
    """, unsafe_allow_html=True)

class APIClient:
    """Enhanced API client with better error handling"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    def _make_request(self, endpoint: str, data: Dict[Any, Any], timeout: int = 300) -> Dict[Any, Any]:
        """Make POST request with enhanced error handling"""
        try:
            with st.spinner('🔄 Processing request...'):
                response = requests.post(
                    f"{self.base_url}{endpoint}", 
                    json=data,
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json()
        except requests.exceptions.Timeout:
            raise Exception(f"⏰ Request timed out after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise Exception("🔌 Could not connect to backend server")
        except requests.exceptions.RequestException as e:
            raise Exception(f"🚨 API request failed: {str(e)}")
    
    def research(self, query: str) -> Dict[Any, Any]:
        return self._make_request("/run/research", {"query": query})
    
    def collect_data(self, query: str) -> Dict[Any, Any]:
        return self._make_request("/run/collect", {"query": query})
    
    def preprocess_data(self, data_path: str, columns: list = None) -> Dict[Any, Any]:
        return self._make_request("/run/preprocess", {
            "data_path": data_path,
            "columns": columns or []
        })
    
    def run_eda(self, data_path: str) -> Dict[Any, Any]:
        return self._make_request("/run/eda", {"data_path": data_path})
    
    def create_visualization(self, query: str, data_path: str) -> Dict[Any, Any]:
        return self._make_request("/run/viz", {
            "query": query,
            "data_path": data_path
        })
    
    def unsloth_finetune(self, model_name: str, domain_name: str) -> Dict[Any, Any]:
        return self._make_request("/run/unsloth", {
            "model_name": model_name,
            "domain_name": domain_name
        }, timeout=1800)  # 30 minutes for fine-tuning
    
    def transformer_finetune(self, model_name: str, task_type: str, dataset: str) -> Dict[Any, Any]:
        return self._make_request("/run/transformer", {
            "ft_model_name": model_name,
            "ft_task_type": task_type,
            "ft_dataset": dataset
        }, timeout=1800)
    
    def health_check(self) -> bool:
        """Check if FastAPI backend is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_tasks(self) -> Dict[Any, Any]:
        """Get all tasks from backend"""
        try:
            response = requests.get(f"{self.base_url}/tasks", timeout=10)
            return response.json() if response.status_code == 200 else {"tasks": []}
        except:
            return {"tasks": []}

@st.cache_resource
def get_api_client():
    """Initialize API client with caching"""
    return APIClient(API_BASE_URL)

def show_header():
    """Display beautiful header"""
    st.markdown("""
    <div class="main-header fade-in-up">
        <h1>🤖 AI Agent Hub</h1>
        <p>Your intelligent multi-agent platform for data science, research & AI</p>
    </div>
    """, unsafe_allow_html=True)

def show_status_dashboard():
    """Show system status dashboard"""
    st.markdown("### 📊 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    client = get_api_client()
    backend_status = client.health_check()
    
    with col1:
        status_color = "success" if backend_status else "error"
        st.markdown(f"""
        <div class="status-card {status_color}">
            <div style="font-size: 1.5rem;">{'🟢' if backend_status else '🔴'}</div>
            <div style="font-weight: 600;">Backend</div>
            <div style="font-size: 0.8rem;">{'Online' if backend_status else 'Offline'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        tasks_data = client.get_tasks()
        total_tasks = len(tasks_data.get("tasks", []))
        st.markdown(f"""
        <div class="status-card">
            <div style="font-size: 1.5rem;">📋</div>
            <div style="font-weight: 600;">{total_tasks}</div>
            <div style="font-size: 0.8rem;">Total Tasks</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        running_tasks = len([t for t in tasks_data.get("tasks", []) if t.get("status") == "running"])
        st.markdown(f"""
        <div class="status-card warning">
            <div style="font-size: 1.5rem;">⚡</div>
            <div style="font-weight: 600;">{running_tasks}</div>
            <div style="font-size: 0.8rem;">Running</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Get file count
        file_count = 0
        if os.path.exists("data/csv_data"):
            file_count = len([f for f in os.listdir("data/csv_data") if f.endswith('.csv')])
        
        st.markdown(f"""
        <div class="status-card">
            <div style="font-size: 1.5rem;">📁</div>
            <div style="font-weight: 600;">{file_count}</div>
            <div style="font-size: 0.8rem;">Data Files</div>
        </div>
        """, unsafe_allow_html=True)
    
    return backend_status

def show_task_selector():
    """Show beautiful task selection cards"""
    st.markdown("### 🎯 Select Your AI Task")
    
    tasks = [
        {
            "id": "research",
            "icon": "🔍",
            "title": "Research & Innovation",
            "description": "Conduct comprehensive research on any topic using AI agents"
        },
        {
            "id": "collect",
            "icon": "📊",
            "title": "Data Collection",
            "description": "Automatically gather and organize data from multiple sources"
        },
        {
            "id": "preprocess",
            "icon": "🧹",
            "title": "Data Preprocessing",
            "description": "Clean, transform and prepare your data for analysis"
        },
        {
            "id": "eda",
            "icon": "📈",
            "title": "Exploratory Analysis",
            "description": "Discover insights and patterns in your datasets"
        },
        {
            "id": "visualize",
            "icon": "📊",
            "title": "Data Visualization",
            "description": "Create stunning charts and interactive visualizations"
        },
        {
            "id": "unsloth",
            "icon": "🔧",
            "title": "Unsloth Fine-tuning",
            "description": "Efficient fine-tuning for large language models"
        },
        {
            "id": "transformer",
            "icon": "🤖",
            "title": "Transformer Fine-tuning",
            "description": "Advanced transformer model training and adaptation"
        }
    ]
    
    # Create task selection grid
    cols = st.columns(3)
    selected_task = None
    
    for i, task in enumerate(tasks):
        with cols[i % 3]:
            if st.button(
                f"{task['icon']} {task['title']}", 
                key=f"task_{task['id']}",
                help=task['description'],
                use_container_width=True
            ):
                selected_task = task['id']
    
    return selected_task

def display_beautiful_result(result: Dict[Any, Any], task_name: str):
    """Display results in a beautiful format"""
    st.markdown(f"### ✅ {task_name} Completed Successfully!")
    
    # Success animation
    st.balloons()
    
    # Results container
    with st.container():
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        # Show key information
        if "result" in result:
            st.markdown("#### 📋 Task Results")
            
            # If result is text, show it nicely formatted
            if isinstance(result["result"], str):
                st.markdown(f"```\n{result['result']}\n```")
            else:
                st.json(result["result"])
        
        # Show inputs used
        if "inputs" in result:
            with st.expander("⚙️ Parameters Used", expanded=False):
                st.json(result["inputs"])
        
        # Show full response
        with st.expander("📄 Full API Response", expanded=False):
            st.json(result)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download option
    if st.button("💾 Save Results", key="save_results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_name.lower().replace(' ', '_')}_results_{timestamp}.json"
        
        # Convert to JSON for download
        json_str = json.dumps(result, indent=2)
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )

def show_file_browser(directory: str, title: str) -> str:
    """Beautiful file browser"""
    st.markdown(f"#### 📁 {title}")
    
    files = []
    if os.path.exists(directory):
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not files:
        st.warning(f"No files found in {directory}")
        return st.text_input("Enter filename manually:", placeholder="filename_without_extension")
    
    # Create file selection with preview
    selected_file = st.selectbox(
        "Choose a file:",
        options=files,
        format_func=lambda x: f"📄 {x}"
    )
    
    if selected_file:
        # Show file info
        file_path = os.path.join(directory, selected_file)
        try:
            df = pd.read_csv(file_path)
            file_size = os.path.getsize(file_path)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Size", f"{file_size/1024:.1f} KB")
            
            # Preview
            with st.expander("👀 Preview", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
                
        except Exception as e:
            st.error(f"Could not preview file: {str(e)}")
    
    return selected_file[:-4] if selected_file else ""

def research_interface():
    """Beautiful research interface"""
    st.markdown("## 🔍 Research & Innovation Hub")
    st.markdown("Harness the power of AI agents to conduct comprehensive research on any topic.")
    
    with st.form("research_form", clear_on_submit=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "🎯 Research Query",
                placeholder="Enter your research question or topic here...\n\nExample: 'Latest trends in artificial intelligence for healthcare'",
                height=120,
                help="Be specific for better results!"
            )
        
        with col2:
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Be specific and detailed
            - Include context or domain
            - Ask focused questions
            - Use keywords effectively
            """)
        
        submit = st.form_submit_button("🚀 Start Research", use_container_width=True)
        
        if submit and query:
            try:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Initializing research agents...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                status_text.text("🌐 Gathering information from sources...")
                progress_bar.progress(50)
                
                client = get_api_client()
                result = client.research(query)
                
                progress_bar.progress(100)
                status_text.text("✅ Research completed!")
                
                display_beautiful_result(result, "Research")
                
            except Exception as e:
                st.error(f"🚨 Research failed: {str(e)}")

def preprocess_interface():
    """Beautiful preprocessing interface"""
    st.markdown("## 🧹 Data Preprocessing Studio")
    st.markdown("Transform and clean your raw data into analysis-ready format.")
    
    with st.form("preprocess_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_file = show_file_browser("data/csv_data", "Raw Data Files")
            
            columns_input = st.text_area(
                "🎯 Columns to Process",
                placeholder="Leave empty to process all columns, or specify:\nname, age, salary, department",
                help="Comma-separated column names",
                height=80
            )
        
        with col2:
            st.markdown("### 🛠️ Preprocessing Options")
            options = st.multiselect(
                "Select operations:",
                ["Remove duplicates", "Handle missing values", "Normalize text", "Convert data types"],
                default=["Remove duplicates", "Handle missing values"]
            )
        
        submit = st.form_submit_button("🧹 Start Preprocessing", use_container_width=True)
        
        if submit and selected_file:
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📊 Loading data...")
                progress_bar.progress(20)
                
                columns = [col.strip() for col in columns_input.split(',') if col.strip()] if columns_input else []
                
                status_text.text("🧹 Preprocessing data...")
                progress_bar.progress(60)
                
                client = get_api_client()
                result = client.preprocess_data(selected_file, columns)
                
                progress_bar.progress(100)
                status_text.text("✅ Preprocessing completed!")
                
                display_beautiful_result(result, "Data Preprocessing")
                
            except Exception as e:
                st.error(f"🚨 Preprocessing failed: {str(e)}")

def main():
    st.set_page_config(
        page_title="AI Agent Hub",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Header
    show_header()
    
    # Status Dashboard
    backend_connected = show_status_dashboard()
    
    if not backend_connected:
        st.error("🔌 Backend server is not available. Please start your FastAPI server.")
        st.code("uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        st.stop()
    
    st.markdown("---")
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        
        page = st.selectbox(
            "Choose your workspace:",
            [
                "🏠 Home",
                "🔍 Research Hub", 
                "📊 Data Collection",
                "🧹 Data Preprocessing",
                "📈 EDA Studio",
                "📊 Visualization Lab",
                "🔧 Model Fine-tuning"
            ],
            key="navigation"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        # Theme selector (placeholder)
        theme = st.selectbox("Theme", ["Modern", "Classic", "Dark"])
        
        # Auto-refresh option
        auto_refresh = st.checkbox("Auto-refresh status", value=True)
        
        if auto_refresh:
            st.markdown("🔄 Auto-refreshing every 30s")
            time.sleep(0.1)  # Small delay for effect
    
    # Main Content Area
    if page == "🏠 Home":
        selected_task = show_task_selector()
        
        if selected_task == "research":
            st.session_state.navigation = "🔍 Research Hub"
            st.rerun()
        elif selected_task == "preprocess":
            st.session_state.navigation = "🧹 Data Preprocessing" 
            st.rerun()
            
    elif page == "🔍 Research Hub":
        research_interface()
        
    elif page == "🧹 Data Preprocessing":
        preprocess_interface()
        
    elif page == "📊 Data Collection":
        st.markdown("## 📊 Data Collection Center")
        st.info("🚧 Data collection interface coming soon!")
        
    elif page == "📈 EDA Studio":
        st.markdown("## 📈 Exploratory Data Analysis Studio")
        st.info("🚧 EDA interface coming soon!")
        
    elif page == "📊 Visualization Lab":
        st.markdown("## 📊 Data Visualization Lab")
        st.info("🚧 Visualization interface coming soon!")
        
    elif page == "🔧 Model Fine-tuning":
        st.markdown("## 🔧 AI Model Fine-tuning Center")
        st.info("🚧 Fine-tuning interface coming soon!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #6b7280;'>
        <p>🤖 <strong>AI Agent Hub</strong> - Powered by FastAPI & Streamlit</p>
        <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Built with ❤️ for the AI community</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()