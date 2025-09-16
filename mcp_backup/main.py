from crewai import Crew, Process
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from agent import AutoMLAgents
from task import AutoMLTasks
import os


app = FastAPI(
    title="My Own Agent",
    version="1.0.0"
)

# Initialize components
agents = AutoMLAgents()
tasks = AutoMLTasks()

# Initialize agents
manager = agents.manager_agent()
data_analyst = agents.data_analyst_agent()
preprocessor = agents.preprocessing_agent()
ml_engineer = agents.ml_engineer_agent()
        

# Request models
class AutoMLRequest(BaseModel):
    file_path: str
    objective: str 
    dataset_name: Optional[str] = None
    

class AutoMLResponse(BaseModel):
    task: str
    status: str
    inputs: Dict[str, Any]
    result: Any
    output_paths: Dict[str, str]


# End points
@app.get("/")
def root():
    return {
        "message": "AutoML API v1.0",
        "endpoints": {
            "full_pipeline": "/run/automl/full",
            "eda_only": "/run/automl/eda",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/run/automl/full", response_model=AutoMLResponse)
def run_full_automl(req: AutoMLRequest):    
    try:
        file_path = f"data/csv_data/{req.file_path}.csv"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Set dataset name
        dataset_name = req.dataset_name or req.file_path

        # Create tasks
        eda_task = tasks.data_analysis_task(data_analyst, file_path, dataset_name)
        preprocessing_task = tasks.preprocessing_task(preprocessor, file_path, dataset_name)
        training_task = tasks.model_training_task(ml_engineer, dataset_name, req.objective)
        optimization_task = tasks.hyperparameter_optimization_task(ml_engineer, dataset_name, req.objective)
        evaluation_task = tasks.final_evaluation_task(ml_engineer, dataset_name, req.objective)
        
        # Create crew
        crew = Crew(
            agents=[manager, data_analyst, preprocessor, ml_engineer],
            tasks=[eda_task, preprocessing_task, training_task, optimization_task, evaluation_task],
            manager_agent=manager,
            memory=True,
            process=Process.hierarchical,
            verbose=True
        )
        
        # Execute pipeline
        result = crew.kickoff(inputs={
            'file_path': file_path,
            'dataset_name': dataset_name,
            'objective': req.objective
        })
        
        # Generate output paths
        output_paths = {
            "eda": f"outputs/AutoML_Output/{dataset_name}_EDA/",
            "preprocessing": f"outputs/AutoML_Output/{dataset_name}_preprocessing/",
            "training": f"outputs/AutoML_Output/{dataset_name}_training/",
            "optimization": f"outputs/AutoML_Output/{dataset_name}_optimization/",
            "evaluation": f"outputs/AutoML_Output/{dataset_name}_final_evaluation/"
        }
        
        return AutoMLResponse(
            task="full_automl_pipeline",
            status="completed",
            inputs=req.model_dump(),
            result=str(result),
            output_paths=output_paths
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@app.post("/run/automl/eda")
def run_eda_only(req: AutoMLRequest):
    try:
        file_path = f"data/csv_data/{req.file_path}.csv"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        dataset_name = req.dataset_name or req.file_path
        
        # Single agent for EDA
        data_analyst = agents.data_analyst_agent()
        eda_task = tasks.data_analysis_task(data_analyst, file_path, dataset_name)
        
        crew = Crew(
            agents=[data_analyst],
            tasks=[eda_task],
            verbose=True
        )
        
        result = crew.kickoff(inputs={
            'file_path': file_path,
            'dataset_name': dataset_name
        })
        
        return {
            "task": "eda_only",
            "status": "completed",
            "inputs": req.model_dump(),
            "result": str(result),
            "output_path": f"outputs/AutoML_Output/{dataset_name}_EDA/"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EDA execution failed: {str(e)}")
