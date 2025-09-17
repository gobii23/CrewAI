from crewai import Crew, Process
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
from crewai_tools import MCPServerAdapter
from agent import AutoMLAgents
from task import AutoMLTasks
import os

app = FastAPI(title="My Own Agent")

# Initialize components
agents = AutoMLAgents()
tasks = AutoMLTasks()

# Initialize agents
manager = agents.manager_agent()
data_analyst = agents.data_analyst_agent()
preprocessor = agents.preprocessing_agent()
ml_engineer = agents.ml_engineer_agent()

# Global pipeline status tracking
pipeline_status = {}


# Request models
class AutoMLRequest(BaseModel):
    file_path: str
    objective: str
    dataset_name: Optional[str] = None
    target_column: Optional[str] = None
    preprocessing_config: Optional[Dict[str, Any]] = None
    ml_model_config: Optional[Dict[str, Any]] = None
    optimization_config: Optional[Dict[str, Any]] = None


class AutoMLResponse(BaseModel):
    task: str
    status: str
    inputs: Dict[str, Any]
    result: Any
    output_paths: Dict[str, str]
    pipeline_id: Optional[str] = None


class PipelineStatus(BaseModel):
    pipeline_id: str
    status: str  # "running", "completed", "failed"
    current_stage: str
    progress: float
    start_time: str
    end_time: Optional[str] = None
    error: Optional[str] = None


# Utility functions
def validate_file_path(file_path: str) -> str:
    """Validate and construct full file path."""
    if not file_path.endswith(".csv"):
        file_path += ".csv"

    full_path = f"data/csv_data/{file_path}"
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"File not found: {full_path}")

    return full_path


def ensure_docker_container():
    """Ensure the AutoML MCP server container is running."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "docker",
                "ps",
                "--format",
                "table {{.Names}}",
                "--filter",
                "name=automl-mcp-server",
            ],
            capture_output=True,
            text=True,
        )
        if "automl-mcp-server" not in result.stdout:
            raise HTTPException(
                status_code=503,
                detail="AutoML MCP server container is not running. Please start it with: docker-compose up -d automl-server",
            )
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Cannot verify AutoML MCP server status: {str(e)}"
        )


# API Routes
@app.get("/")
def root():
    return {
        "message": "My Own Agent",
        "description": "Machine Learning pipeline Automation",
        "endpoints": {
            "full_pipeline": "/run/automl/full",
            "eda_only": "/run/automl/eda",
            "preprocessing_only": "/run/automl/preprocessing",
            "training_only": "/run/automl/training",
            "optimization_only": "/run/automl/optimization",
            "evaluation_only": "/run/automl/evaluation",
            "pipeline_status": "/status/{pipeline_id}",
            "health": "/health",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        ensure_docker_container()
        return {
            "status": "healthy",
            "automl_server": "running",
            "timestamp": pipeline_status.get("last_check", "unknown"),
        }
    except HTTPException as e:
        return {
            "status": "unhealthy",
            "error": e.detail,
            "automl_server": "not_available",
        }


@app.get("/status/{pipeline_id}")
def get_pipeline_status(pipeline_id: str):
    """Get status of a running pipeline."""
    if pipeline_id not in pipeline_status:
        raise HTTPException(status_code=404, detail="Pipeline ID not found")

    return pipeline_status[pipeline_id]


@app.post("/run/automl/full", response_model=AutoMLResponse)
def run_full_automl(req: AutoMLRequest, background_tasks: BackgroundTasks):
    """Run complete AutoML pipeline with all stages."""
    try:
        ensure_docker_container()

        # Validate inputs
        file_path = validate_file_path(req.file_path)
        dataset_name = req.dataset_name or req.file_path.replace(".csv", "")

        if not req.target_column:
            raise HTTPException(
                status_code=400,
                detail="target_column is required for full AutoML pipeline",
            )

        # Create pipeline ID
        import uuid, datetime

        pipeline_id = str(uuid.uuid4())[:8]

        # Initialize pipeline status
        pipeline_status[pipeline_id] = {
            "pipeline_id": pipeline_id,
            "status": "running",
            "current_stage": "initialization",
            "progress": 0.0,
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": None,
            "error": None,
        }

        def run_pipeline():
            try:
                # Update status
                pipeline_status[pipeline_id]["current_stage"] = "data_analysis"
                pipeline_status[pipeline_id]["progress"] = 10.0

                # Use MCP Server with context manager
                with MCPServerAdapter(tasks.server_params) as mcp_tools:
                    # Create tasks with proper dependencies
                    eda_task = tasks.data_analysis_task(
                        data_analyst, file_path, dataset_name
                    )

                    pipeline_status[pipeline_id]["current_stage"] = "preprocessing"
                    pipeline_status[pipeline_id]["progress"] = 25.0

                    preprocessing_task = tasks.preprocessing_task(
                        preprocessor, file_path, dataset_name
                    )
                    preprocessing_task.context = [eda_task]  # Depend on EDA

                    pipeline_status[pipeline_id]["current_stage"] = "model_training"
                    pipeline_status[pipeline_id]["progress"] = 45.0

                    training_task = tasks.model_training_task(
                        ml_engineer, dataset_name, req.objective
                    )
                    training_task.context = [
                        preprocessing_task
                    ]  # Depend on preprocessing

                    pipeline_status[pipeline_id]["current_stage"] = "optimization"
                    pipeline_status[pipeline_id]["progress"] = 65.0

                    optimization_task = tasks.hyperparameter_optimization_task(
                        ml_engineer, dataset_name, req.objective
                    )
                    optimization_task.context = [training_task]  # Depend on training

                    pipeline_status[pipeline_id]["current_stage"] = "final_evaluation"
                    pipeline_status[pipeline_id]["progress"] = 85.0

                    evaluation_task = tasks.final_evaluation_task(
                        ml_engineer, dataset_name, req.objective
                    )
                    evaluation_task.context = [
                        optimization_task
                    ]  # Depend on optimization

                    # Add MCP tools to agents
                    data_analyst.tools = mcp_tools
                    preprocessor.tools = mcp_tools
                    ml_engineer.tools = mcp_tools

                    # Create crew with hierarchical process
                    crew = Crew(
                        agents=[data_analyst, preprocessor, ml_engineer],
                        tasks=[
                            eda_task,
                            preprocessing_task,
                            training_task,
                            optimization_task,
                            evaluation_task,
                        ],
                        manager_agent=manager,
                        memory=False,
                        process=Process.hierarchical,
                        verbose=True,
                    )

                    # Execute pipeline
                    result = crew.kickoff(
                        inputs={
                            "file_path": file_path,
                            "dataset_name": dataset_name,
                            "objective": req.objective,
                            "target_column": req.target_column,
                            "preprocessing_config": req.preprocessing_config or {},
                            "ml_model_config": req.ml_model_config
                            or {},  # Updated reference
                            "optimization_config": req.optimization_config or {},
                        }
                    )

                    # Update completion status
                    pipeline_status[pipeline_id].update(
                        {
                            "status": "completed",
                            "current_stage": "completed",
                            "progress": 100.0,
                            "end_time": datetime.datetime.now().isoformat(),
                            "result": str(result),
                        }
                    )

            except Exception as e:
                pipeline_status[pipeline_id].update(
                    {
                        "status": "failed",
                        "current_stage": "error",
                        "progress": 0.0,
                        "end_time": datetime.datetime.now().isoformat(),
                        "error": str(e),
                    }
                )

        # Run pipeline in background
        background_tasks.add_task(run_pipeline)

        # Generate output paths
        output_paths = {
            "eda": f"outputs/AutoML_Output/{dataset_name}_EDA/",
            "preprocessing": f"outputs/AutoML_Output/{dataset_name}_preprocessing/",
            "training": f"outputs/AutoML_Output/{dataset_name}_training/",
            "optimization": f"outputs/AutoML_Output/{dataset_name}_optimization/",
            "evaluation": f"outputs/AutoML_Output/{dataset_name}_final_evaluation/",
        }

        return AutoMLResponse(
            task="full_automl_pipeline",
            status="started",
            inputs=req.model_dump(),
            result="Pipeline started successfully. Use /status/{pipeline_id} to track progress.",
            output_paths=output_paths,
            pipeline_id=pipeline_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Pipeline initialization failed: {str(e)}"
        )


@app.post("/run/automl/eda")
def run_eda_only(req: AutoMLRequest):
    """Run exploratory data analysis only."""
    try:
        ensure_docker_container()

        file_path = validate_file_path(req.file_path)
        dataset_name = req.dataset_name or req.file_path.replace(".csv", "")

        # Use MCP Server with context manager for EDA only
        with MCPServerAdapter(tasks.server_params) as mcp_tools:
            # Create EDA agent and task
            data_analyst_agent = agents.data_analyst_agent()
            data_analyst_agent.tools = mcp_tools

            eda_task = tasks.data_analysis_task(
                data_analyst_agent, file_path, dataset_name
            )

            # Create crew with single agent
            crew = Crew(
                agents=[data_analyst_agent],
                tasks=[eda_task],
                manager_agent=manager,
                verbose=True,
            )

            # Execute EDA
            result = crew.kickoff(
                inputs={"file_path": file_path, "dataset_name": dataset_name}
            )

        return AutoMLResponse(
            task="eda_only",
            status="completed",
            inputs=req.model_dump(),
            result=str(result),
            output_paths={"eda": f"outputs/AutoML_Output/{dataset_name}_EDA/"},
            pipeline_id=None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EDA execution failed: {str(e)}")


@app.post("/run/automl/preprocessing")
def run_preprocessing_only(req: AutoMLRequest):
    """Run data preprocessing only."""
    try:
        ensure_docker_container()

        file_path = validate_file_path(req.file_path)
        dataset_name = req.dataset_name or req.file_path.replace(".csv", "")

        if not req.target_column:
            raise HTTPException(
                status_code=400, detail="target_column is required for preprocessing"
            )

        with MCPServerAdapter(tasks.server_params) as mcp_tools:
            preprocessor_agent = agents.preprocessing_agent()
            preprocessor_agent.tools = mcp_tools

            preprocessing_task = tasks.preprocessing_task(
                preprocessor_agent, file_path, dataset_name
            )

            crew = Crew(
                agents=[preprocessor_agent], tasks=[preprocessing_task], verbose=True
            )

            result = crew.kickoff(
                inputs={
                    "file_path": file_path,
                    "dataset_name": dataset_name,
                    "target_column": req.target_column,
                    "preprocessing_config": req.preprocessing_config or {},
                }
            )

        return AutoMLResponse(
            task="preprocessing_only",
            status="completed",
            inputs=req.model_dump(),
            result=str(result),
            output_paths={
                "preprocessing": f"outputs/AutoML_Output/{dataset_name}_preprocessing/"
            },
            pipeline_id=None,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Preprocessing execution failed: {str(e)}"
        )


@app.post("/run/automl/training")
def run_training_only(req: AutoMLRequest):
    """Run model training only."""
    try:
        ensure_docker_container()

        dataset_name = req.dataset_name or req.file_path.replace(".csv", "")

        with MCPServerAdapter(tasks.server_params) as mcp_tools:
            ml_engineer_agent = agents.ml_engineer_agent()
            ml_engineer_agent.tools = mcp_tools

            training_task = tasks.model_training_task(
                ml_engineer_agent, dataset_name, req.objective
            )

            crew = Crew(agents=[ml_engineer_agent], tasks=[training_task], verbose=True)

            result = crew.kickoff(
                inputs={
                    "dataset_name": dataset_name,
                    "objective": req.objective,
                    "ml_model_config": req.ml_model_config or {},  # Updated reference
                }
            )

        return AutoMLResponse(
            task="training_only",
            status="completed",
            inputs=req.model_dump(),
            result=str(result),
            output_paths={
                "training": f"outputs/AutoML_Output/{dataset_name}_training/"
            },
            pipeline_id=None,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Training execution failed: {str(e)}"
        )


@app.post("/run/automl/optimization")
def run_optimization_only(req: AutoMLRequest):
    """Run hyperparameter optimization only."""
    try:
        ensure_docker_container()

        dataset_name = req.dataset_name or req.file_path.replace(".csv", "")

        with MCPServerAdapter(tasks.server_params) as mcp_tools:
            ml_engineer_agent = agents.ml_engineer_agent()
            ml_engineer_agent.tools = mcp_tools

            optimization_task = tasks.hyperparameter_optimization_task(
                ml_engineer_agent, dataset_name, req.objective
            )

            crew = Crew(
                agents=[ml_engineer_agent], tasks=[optimization_task], verbose=True
            )

            result = crew.kickoff(
                inputs={
                    "dataset_name": dataset_name,
                    "objective": req.objective,
                    "optimization_config": req.optimization_config or {},
                }
            )

        return AutoMLResponse(
            task="optimization_only",
            status="completed",
            inputs=req.model_dump(),
            result=str(result),
            output_paths={
                "optimization": f"outputs/AutoML_Output/{dataset_name}_optimization/"
            },
            pipeline_id=None,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Optimization execution failed: {str(e)}"
        )


@app.post("/run/automl/evaluation")
def run_evaluation_only(req: AutoMLRequest):
    """Run final model evaluation only."""
    try:
        ensure_docker_container()

        dataset_name = req.dataset_name or req.file_path.replace(".csv", "")

        with MCPServerAdapter(tasks.server_params) as mcp_tools:
            ml_engineer_agent = agents.ml_engineer_agent()
            ml_engineer_agent.tools = mcp_tools

            evaluation_task = tasks.final_evaluation_task(
                ml_engineer_agent, dataset_name, req.objective
            )

            crew = Crew(
                agents=[ml_engineer_agent], tasks=[evaluation_task], verbose=True
            )

            result = crew.kickoff(
                inputs={"dataset_name": dataset_name, "objective": req.objective}
            )

        return AutoMLResponse(
            task="evaluation_only",
            status="completed",
            inputs=req.model_dump(),
            result=str(result),
            output_paths={
                "evaluation": f"outputs/AutoML_Output/{dataset_name}_final_evaluation/"
            },
            pipeline_id=None,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Evaluation execution failed: {str(e)}"
        )
