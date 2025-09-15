
import os, re
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import uuid

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Crew, Process
from agents import Own_Agents
from tasks import Own_Tasks

# ==== TOOLS ====
from crewai_tools import SerperDevTool
from tools.plot_tool import PlotTool
from tools.eda_tool import EDATool
from tools.csv_rag_tool import CsvRAGTool
from tools.terminal_tool import TerminalTool
from tools.fine_tuning_unsloth_tool import FineTuningTool
from tools.fine_tuning_transformer_tool import FTTool
from tools.data_preprocessing_tool import DataPreprocessingTool

# Initialize tools
search_tool = SerperDevTool()
terminal_tool = TerminalTool()
eda_tool = EDATool()
plot_tool = PlotTool()
preprocessing_tool = DataPreprocessingTool()
fine_tuning_tool = FineTuningTool()
ft_tool = FTTool()

app = FastAPI(
    title="AI Agent API",
    description="Multi-agent AI system for data processing and ML tasks",
    version="2.0.0"
)

# Add CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Task Management ====
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskInfo(BaseModel):
    task_id: str
    task_type: str
    status: TaskStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: Optional[str] = None

# In-memory task storage (use Redis in production)
tasks_storage: Dict[str, TaskInfo] = {}

# ==== Input Models ====
class QueryRequest(BaseModel):
    query: str

class VisualRequest(BaseModel):
    query: str
    data_path: str

class EDARequest(BaseModel):
    data_path: str

class PreprocessRequest(BaseModel):
    data_path: str
    columns: List[str] = []

class FineTuningRequest(BaseModel):
    model_name: str
    domain_name: str

class TransformerFTRequest(BaseModel):
    ft_model_name: str
    ft_task_type: str
    ft_dataset: str

class AsyncTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

# ==== Agents (Initialize once) ====
agents = Own_Agents()
gopinath_agent = agents.gopinath_agent()
researcher_agent = agents.researcher_agent()
data_collector_agent = agents.data_collector_agent()
data_preprocessor_agent = agents.data_preprocessor_agent()
eda_agent = agents.eda_agent()
finetuning_agent = agents.unsloth_finetuning_agent()
ft_agent = agents.transformer_finetuning_agent()

tasks = Own_Tasks()

# ==== Helper Functions ====
def create_task(task_type: str) -> str:
    """Create a new task and return task ID"""
    task_id = str(uuid.uuid4())
    task_info = TaskInfo(
        task_id=task_id,
        task_type=task_type,
        status=TaskStatus.PENDING,
        created_at=datetime.now()
    )
    tasks_storage[task_id] = task_info
    return task_id

def update_task_status(task_id: str, status: TaskStatus, result: Any = None, error: str = None, progress: str = None):
    """Update task status"""
    if task_id in tasks_storage:
        task_info = tasks_storage[task_id]
        task_info.status = status
        task_info.result = result
        task_info.error = error
        task_info.progress = progress
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            task_info.completed_at = datetime.now()

async def run_crew_task(crew: Crew, inputs: Dict[str, Any], task_id: str):
    """Run CrewAI task asynchronously"""
    try:
        update_task_status(task_id, TaskStatus.RUNNING, progress="Starting task...")
        
        # Run the crew task (this is synchronous, so we run it in a thread)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, crew.kickoff, inputs)
        
        update_task_status(task_id, TaskStatus.COMPLETED, result=result)
        return result
    except Exception as e:
        update_task_status(task_id, TaskStatus.FAILED, error=str(e))
        raise e

# ==== API Endpoints ====

@app.get("/")
async def root():
    return {
        "message": "AI Agent API", 
        "version": "2.0.0",
        "docs": "/docs",
        "active_tasks": len([t for t in tasks_storage.values() if t.status == TaskStatus.RUNNING])
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now(),
        "total_tasks": len(tasks_storage),
        "running_tasks": len([t for t in tasks_storage.values() if t.status == TaskStatus.RUNNING])
    }

@app.get("/tasks")
async def list_tasks():
    """List all tasks with their status"""
    return {"tasks": list(tasks_storage.values())}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_storage[task_id]

# ==== Synchronous Endpoints (Original behavior) ====

@app.post("/run/viz")
async def run_viz(req: VisualRequest):
    """Synchronous visualization task"""
    try:
        csv_path = f"data/csv_cleaned_data/{req.data_path}_cleaned.csv"
        rag_tool = CsvRAGTool(file_path=csv_path)
        viz_task = tasks.data_visualization_task(gopinath_agent, rag_tool, plot_tool, req.query)
        viz_crew = Crew(agents=[gopinath_agent], tasks=[viz_task], process=Process.sequential)
        result = viz_crew.kickoff(inputs={"query": req.query, "file_path": csv_path})
        return {"task": "viz", "inputs": req.model_dump(), "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run/research")
async def run_research(req: QueryRequest):
    """Synchronous research task"""
    try:
        safe_filename = re.sub(r"[^a-zA-Z0-9_\-]", "_", req.query.strip().lower())[:50]
        summary_filename = f"output/summary/summary_{safe_filename}.md"
        os.makedirs("output/summary", exist_ok=True)
        research_task = tasks.research_innovation_task(researcher_agent, search_tool, req.query, summary_filename)
        research_task.output_file = summary_filename
        research_crew = Crew(agents=[researcher_agent], tasks=[research_task], process=Process.sequential)
        result = research_crew.kickoff(inputs={"query": req.query})
        return {"task": "research", "query": req.query, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run/collect")
async def run_collect(req: QueryRequest):
    """Synchronous data collection task"""
    try:
        collect_task = tasks.data_collection_task(data_collector_agent, search_tool, terminal_tool, req.query)
        collect_crew = Crew(agents=[data_collector_agent], tasks=[collect_task], process=Process.sequential)
        result = collect_crew.kickoff(inputs={"query": req.query})
        return {"task": "collect", "query": req.query, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run/preprocess")
async def run_preprocess(req: PreprocessRequest):
    """Synchronous preprocessing task"""
    try:
        file_path = f"data/csv_data/{req.data_path}.csv"
        preprocess_task = tasks.data_preprocessing_task(data_preprocessor_agent, preprocessing_tool, file_path, req.columns)
        preprocess_crew = Crew(agents=[data_preprocessor_agent], tasks=[preprocess_task], process=Process.sequential)
        result = preprocess_crew.kickoff(inputs={"file_path": file_path, "selected_columns": req.columns})
        return {"task": "preprocess", "inputs": req.model_dump(), "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run/eda")
async def run_eda(req: EDARequest):
    """Synchronous EDA task"""
    try:
        csv_path = f"data/csv_data/{req.data_path}.csv"
        eda_task = tasks.eda_task(eda_agent, eda_tool, csv_path)
        eda_crew = Crew(agents=[eda_agent], tasks=[eda_task], process=Process.sequential)
        result = eda_crew.kickoff(inputs={"csv_path": csv_path})
        return {"task": "eda", "inputs": req.model_dump(), "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run/unsloth")
async def run_unsloth(req: FineTuningRequest):
    """Synchronous Unsloth fine-tuning task"""
    try:
        unsloth_task = tasks.unsloth_fine_tuning_task(finetuning_agent, search_tool, fine_tuning_tool, req.model_name, req.domain_name)
        unsloth_crew = Crew(agents=[finetuning_agent], tasks=[unsloth_task], process=Process.sequential)
        result = unsloth_crew.kickoff(inputs={"model_name": req.model_name, "domain_name": req.domain_name})
        return {"task": "unsloth", "inputs": req.model_dump(), "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run/transformer")
async def run_transformer(req: TransformerFTRequest):
    """Synchronous transformer fine-tuning task"""
    try:
        transformer_task = tasks.transformer_fine_tuning_task(ft_agent, ft_tool, req.ft_model_name, req.ft_task_type, req.ft_dataset)
        transformer_crew = Crew(agents=[ft_agent], tasks=[transformer_task], process=Process.sequential)
        result = transformer_crew.kickoff(inputs={
            "ft_model_name": req.ft_model_name,
            "ft_task_type": req.ft_task_type,
            "ft_dataset": req.ft_dataset
        })
        return {"task": "transformer", "inputs": req.model_dump(), "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==== Asynchronous Endpoints (New) ====

@app.post("/async/research", response_model=AsyncTaskResponse)
async def async_research(req: QueryRequest, background_tasks: BackgroundTasks):
    """Start asynchronous research task"""
    task_id = create_task("research")
    
    async def research_background():
        try:
            safe_filename = re.sub(r"[^a-zA-Z0-9_\-]", "_", req.query.strip().lower())[:50]
            summary_filename = f"output/summary/summary_{safe_filename}.md"
            os.makedirs("output/summary", exist_ok=True)
            research_task = tasks.research_innovation_task(researcher_agent, search_tool, req.query, summary_filename)
            research_task.output_file = summary_filename
            research_crew = Crew(agents=[researcher_agent], tasks=[research_task], process=Process.sequential)
            await run_crew_task(research_crew, {"query": req.query}, task_id)
        except Exception as e:
            update_task_status(task_id, TaskStatus.FAILED, error=str(e))
    
    background_tasks.add_task(research_background)
    return AsyncTaskResponse(
        task_id=task_id,
        status="started",
        message="Research task started in background"
    )

@app.post("/async/unsloth", response_model=AsyncTaskResponse)
async def async_unsloth(req: FineTuningRequest, background_tasks: BackgroundTasks):
    """Start asynchronous Unsloth fine-tuning task"""
    task_id = create_task("unsloth_finetuning")
    
    async def unsloth_background():
        try:
            unsloth_task = tasks.unsloth_fine_tuning_task(
                finetuning_agent, search_tool, fine_tuning_tool, req.model_name, req.domain_name
            )
            unsloth_crew = Crew(agents=[finetuning_agent], tasks=[unsloth_task], process=Process.sequential)
            await run_crew_task(unsloth_crew, {
                "model_name": req.model_name, 
                "domain_name": req.domain_name
            }, task_id)
        except Exception as e:
            update_task_status(task_id, TaskStatus.FAILED, error=str(e))
    
    background_tasks.add_task(unsloth_background)
    return AsyncTaskResponse(
        task_id=task_id,
        status="started",
        message="Unsloth fine-tuning started in background"
    )

@app.post("/async/transformer", response_model=AsyncTaskResponse)
async def async_transformer(req: TransformerFTRequest, background_tasks: BackgroundTasks):
    """Start asynchronous transformer fine-tuning task"""
    task_id = create_task("transformer_finetuning")
    
    async def transformer_background():
        try:
            transformer_task = tasks.transformer_fine_tuning_task(
                ft_agent, ft_tool, req.ft_model_name, req.ft_task_type, req.ft_dataset
            )
            transformer_crew = Crew(agents=[ft_agent], tasks=[transformer_task], process=Process.sequential)
            await run_crew_task(transformer_crew, {
                "ft_model_name": req.ft_model_name,
                "ft_task_type": req.ft_task_type,
                "ft_dataset": req.ft_dataset
            }, task_id)
        except Exception as e:
            update_task_status(task_id, TaskStatus.FAILED, error=str(e))
    
    background_tasks.add_task(transformer_background)
    return AsyncTaskResponse(
        task_id=task_id,
        status="started",
        message="Transformer fine-tuning started in background"
    )

@app.post("/async/collect", response_model=AsyncTaskResponse)
async def async_collect(req: QueryRequest, background_tasks: BackgroundTasks):
    """Start asynchronous data collection task"""
    task_id = create_task("data_collection")
    
    async def collect_background():
        try:
            collect_task = tasks.data_collection_task(data_collector_agent, search_tool, terminal_tool, req.query)
            collect_crew = Crew(agents=[data_collector_agent], tasks=[collect_task], process=Process.sequential)
            await run_crew_task(collect_crew, {"query": req.query}, task_id)
        except Exception as e:
            update_task_status(task_id, TaskStatus.FAILED, error=str(e))
    
    background_tasks.add_task(collect_background)
    return AsyncTaskResponse(
        task_id=task_id,
        status="started",
        message="Data collection started in background"
    )

# ==== Utility Endpoints ====

@app.get("/files/csv")
async def list_csv_files():
    """List available CSV files"""
    files = {"raw": [], "cleaned": []}
    
    # Raw CSV files
    if os.path.exists("data/csv_data"):
        files["raw"] = [f for f in os.listdir("data/csv_data") if f.endswith('.csv')]
    
    # Cleaned CSV files
    if os.path.exists("data/csv_cleaned_data"):
        files["cleaned"] = [f for f in os.listdir("data/csv_cleaned_data") if f.endswith('.csv')]
    
    return files

@app.get("/files/summaries")
async def list_summaries():
    """List available research summaries"""
    summaries = []
    if os.path.exists("output/summary"):
        summaries = [f for f in os.listdir("output/summary") if f.endswith('.md')]
    return {"summaries": summaries}

@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel/remove a task"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = tasks_storage[task_id]
    if task_info.status == TaskStatus.RUNNING:
        # In production, you'd want to actually cancel the running task
        task_info.status = TaskStatus.FAILED
        task_info.error = "Task cancelled by user"
        task_info.completed_at = datetime.now()
    
    del tasks_storage[task_id]
    return {"message": f"Task {task_id} removed"}

@app.delete("/tasks")
async def clear_completed_tasks():
    """Clear all completed/failed tasks"""
    completed_tasks = [
        task_id for task_id, task_info in tasks_storage.items() 
        if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
    ]
    
    for task_id in completed_tasks:
        del tasks_storage[task_id]
    
    return {"message": f"Cleared {len(completed_tasks)} completed tasks"}

# ==== WebSocket for Real-time Updates (Optional) ====
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ==== Startup Event ====
@app.on_event("startup")
async def startup_event():
    """Initialize required directories on startup"""
    directories = [
        "data/csv_data",
        "data/csv_cleaned_data", 
        "output/summary",
        "output/visualizations",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("🚀 AI Agent API started successfully!")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
