#!/usr/bin/env python3
"""
Setup script for running the hybrid AI Agent application
- FastAPI backend
- Streamlit frontend
"""

import subprocess
import sys
import time
import os
from pathlib import Path
import threading

def install_requirements():
    """Install required packages"""
    requirements = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0", 
        "streamlit==1.28.1",
        "requests==2.31.0",
        "pandas==2.1.3",
        "pydantic==2.4.2",
        # Add your existing requirements here
        "crewai",
        "crewai-tools",
        # Add other dependencies as needed
    ]
    
    print("📦 Installing requirements...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ Installed: {req}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")

def create_directory_structure():
    """Create required directory structure"""
    directories = [
        "data/csv_data",
        "data/csv_cleaned_data",
        "output/summary", 
        "output/visualizations",
        "logs",
        "models",
        "config"
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def run_fastapi():
    """Run FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    try:
        subprocess.run([
            "uvicorn", "main:app", 
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 FastAPI backend stopped")

def run_streamlit():
    """Run Streamlit frontend"""
    print("🌐 Starting Streamlit frontend...")
    time.sleep(3)  # Wait for FastAPI to start
    try:
        subprocess.run([
            "streamlit", "run", "streamlit_frontend.py",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit frontend stopped")

def check_files():
    """Check if required files exist"""
    required_files = [
        "main.py",  # FastAPI backend
        "streamlit_frontend.py",  # Streamlit frontend
        "agents.py",  # Your agents
        "tasks.py"   # Your tasks
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all files are in the current directory.")
        return False
    
    print("✅ All required files found")
    return True

def main():
    print("🤖 AI Agent Hybrid App Setup")
    print("=" * 40)
    
    # Check files
    if not check_files():
        return
    
    # Install requirements
    install_requirements()
    
    # Create directories
    create_directory_structure()
    
    print("\n🎯 Setup complete!")
    print("\n" + "=" * 40)
    print("🚀 STARTING SERVICES")
    print("=" * 40)
    
    # Start both services
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    
    try:
        # Start FastAPI in background
        fastapi_thread.start()
        print("✅ FastAPI started on: http://localhost:8000")
        print("📖 API Docs available at: http://localhost:8000/docs")
        
        # Start Streamlit
        streamlit_thread.start()
        print("✅ Streamlit started on: http://localhost:8501")
        
        # Keep main thread alive
        print("\n🎉 Both services are running!")
        print("📝 Press Ctrl+C to stop both services")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        sys.exit(0)

if __name__ == "__main__":
    main()