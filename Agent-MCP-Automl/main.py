from crewai import Crew, Process
from crewai_tools import MCPServerAdapter
from dotenv import load_dotenv
from agent import AutoMLAgents
from task import AutoMLIndividualTasks
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from config import server_params
from fastapi import File, UploadFile
import shutil
import os

load_dotenv()

app = FastAPI(title="AutoML Pipeline API")


class DataInfoRequest(BaseModel):
    file_name: str
    dataset_name: Optional[str] = "dataset"


class CSVReadRequest(BaseModel):
    file_name: str
    dataset_name: Optional[str] = "dataset"


class PreprocessingRequest(BaseModel):
    file_name: str
    target_column: str
    dataset_name: Optional[str] = "dataset"


class DataPrepareRequest(BaseModel):
    file_name: str
    target_column: str
    problem_type: str
    dataset_name: Optional[str] = "dataset"


class ModelTrainingRequest(BaseModel):
    problem_type: str
    file_name: str
    target_column: str
    dataset_name: Optional[str] = "dataset"


class VisualizationRequest(BaseModel):
    file_name: str
    target_column: Optional[str] = None
    dataset_name: Optional[str] = "dataset"


class HyperparameterRequest(BaseModel):
    model_name: str
    file_name: str
    target_column: str
    problem_type: str
    n_trials: int = 100
    scoring: str = "auto"
    random_state: int = 42
    dataset_name: Optional[str] = "dataset"


class PredictionRequest(BaseModel):
    model_name: str
    file_name: str
    target_column: str
    problem_type: str
    input_data: str
    n_trials: int = 100
    scoring: str = "auto"
    random_state: int = 42
    dataset_name: Optional[str] = "dataset"


class ExternalTestRequest(BaseModel):
    main_file_name: str
    target_column: str
    problem_type: str
    test_file_name: str
    dataset_name: Optional[str] = "dataset"


class FeatureImportanceRequest(BaseModel):
    file_name: str
    target_column: str
    problem_type: str
    dataset_name: Optional[str] = "dataset"


class CompletePipelineRequest(BaseModel):
    file_name: str
    target_column: str
    problem_type: str
    dataset_name: Optional[str] = "dataset"


@app.get("/")
def read_root():
    return {"message": "AutoML Pipeline API", "status": "running"}


@app.get("/tools")
def list_tools():
    """List available MCP tools"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            return {
                "tools": [
                    {"name": tool.name, "description": getattr(tool, "description", "")}
                    for tool in tools
                ],
                "count": len(tools)
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    

@app.post("/upload/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset file"""
    try:
        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "status": "success",
            "filename": file.filename,
            "file_path": file_path,
            "message": f"File uploaded successfully. Use filename: {file.filename}"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/data/info")
def get_data_info(request: DataInfoRequest):
    """Get comprehensive dataset information"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.data_analyst_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.information_about_data_task(
                agent=agent,
                file_name=request.file_name,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "file_name": request.file_name
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/data/read-csv")
def read_csv_data(request: CSVReadRequest):
    """Load and validate CSV dataset"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.data_analyst_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.reading_csv_task(
                agent=agent,
                file_name=request.file_name,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "file_name": request.file_name
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/data/preprocessing")
def preprocess_data(request: PreprocessingRequest):
    """Execute comprehensive data preprocessing"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.preprocessing_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.preprocessing_data_task(
                agent=agent,
                file_name=request.file_name,
                target_column=request.target_column,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "target_column": request.target_column
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/data/prepare")
def prepare_data(request: DataPrepareRequest):
    """Prepare data for model training with encoding and scaling"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.preprocessing_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.prepare_data_task(
                agent=agent,
                file_name=request.file_name,
                target_column=request.target_column,
                problem_type=request.problem_type,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "problem_type": request.problem_type
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/visualize/correlation-numerical")
def visualize_numerical_correlation(request: VisualizationRequest):
    """Generate numerical correlation matrix visualization"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.data_analyst_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.visualize_correlation_numerical_task(
                agent=agent,
                file_name=request.file_name,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/visualize/correlation-categorical")
def visualize_categorical_correlation(request: VisualizationRequest):
    """Generate categorical correlation matrix visualization"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.data_analyst_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.visualize_correlation_categorical_task(
                agent=agent,
                file_name=request.file_name,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/visualize/correlation-final")
def visualize_final_correlation(request: VisualizationRequest):
    """Generate final correlation matrix after preprocessing"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.data_analyst_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.visualize_correlation_final_task(
                agent=agent,
                file_name=request.file_name,
                target_column=request.target_column,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "target_column": request.target_column
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/visualize/outliers")
def visualize_outliers(request: VisualizationRequest):
    """Generate outlier detection visualizations"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.data_analyst_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.visualize_outliers_task(
                agent=agent,
                file_name=request.file_name,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/visualize/outliers-final")
def visualize_final_outliers(request: VisualizationRequest):
    """Generate outlier visualizations after preprocessing"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.data_analyst_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.visualize_outliers_final_task(
                agent=agent,
                file_name=request.file_name,
                target_column=request.target_column,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "target_column": request.target_column
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/visualize/accuracy-matrix")
def visualize_accuracy_matrix(request: VisualizationRequest):
    """Generate accuracy/confusion matrix visualizations"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.ml_engineer_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.visualize_accuracy_matrix_task(
                agent=agent,
                file_name=request.file_name,
                target_column=request.target_column,
                problem_type="auto",
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "target_column": request.target_column
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/models/train")
def train_models(request: ModelTrainingRequest):
    """Train and evaluate multiple ML models"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.ml_engineer_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.models_training_task(
                agent=agent,
                problem_type=request.problem_type,
                file_name=request.file_name,
                target_column=request.target_column,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "problem_type": request.problem_type,
                "target_column": request.target_column
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/models/hyperparameter-optimization")
def optimize_hyperparameters(request: HyperparameterRequest):
    """Optimize hyperparameters for the best model"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.ml_engineer_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.best_model_hyperparameter_task(
                agent=agent,
                model_name=request.model_name,
                file_name=request.file_name,
                target_column=request.target_column,
                problem_type=request.problem_type,
                tools=tools,
                n_trials=request.n_trials,
                scoring=request.scoring,
                random_state=request.random_state,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "optimization_params": {
                    "model_name": request.model_name,
                    "n_trials": request.n_trials,
                    "scoring": request.scoring
                }
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/models/predict")
def predict_values(request: PredictionRequest):
    """Generate predictions for new input data"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.ml_engineer_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.predict_value_task(
                agent=agent,
                model_name=request.model_name,
                file_name=request.file_name,
                target_column=request.target_column,
                problem_type=request.problem_type,
                input_data=request.input_data,
                tools=tools,
                n_trials=request.n_trials,
                scoring=request.scoring,
                random_state=request.random_state,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "input_data": request.input_data,
                "model_used": request.model_name
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/models/test-external")
def test_external_data(request: ExternalTestRequest):
    """Test model on external dataset"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.ml_engineer_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.test_external_data_task(
                agent=agent,
                main_file_name=request.main_file_name,
                target_column=request.target_column,
                problem_type=request.problem_type,
                test_file_name=request.test_file_name,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "main_file": request.main_file_name,
                "test_file": request.test_file_name
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/features/importance")
def analyze_feature_importance(request: FeatureImportanceRequest):
    """Analyze feature importance using XGBoost"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.ml_engineer_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.feature_importance_analysis_task(
                agent=agent,
                file_name=request.file_name,
                target_column=request.target_column,
                problem_type=request.problem_type,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "target_column": request.target_column,
                "problem_type": request.problem_type
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/pipeline/complete")
def run_complete_pipeline(request: CompletePipelineRequest):
    """Execute complete AutoML pipeline end-to-end"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            agent = agents.ml_engineer_agent()
            
            tasks = AutoMLIndividualTasks()
            task = tasks.complete_automl_pipeline_task(
                agent=agent,
                file_name=request.file_name,
                target_column=request.target_column,
                problem_type=request.problem_type,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "problem_type": request.problem_type,
                "target_column": request.target_column,
                "pipeline_status": "completed"
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/pipeline/sequential")
def run_sequential_pipeline(request: CompletePipelineRequest):
    """Execute AutoML pipeline with task chaining"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            agents = AutoMLAgents()
            data_analyst = agents.data_analyst_agent()
            preprocessor = agents.preprocessing_agent()
            ml_engineer = agents.ml_engineer_agent()
            
            tasks_factory = AutoMLIndividualTasks()
            
            # Task 1: Data Information
            data_info_task = tasks_factory.information_about_data_task(
                agent=data_analyst,
                file_name=request.file_name,
                tools=tools,
                dataset_name=request.dataset_name
            )
            
            # Task 2: Data Preprocessing
            preprocessing_task = tasks_factory.preprocessing_data_task(
                agent=preprocessor,
                file_name=request.file_name,
                target_column=request.target_column,
                tools=tools,
                dataset_name=request.dataset_name
            )
            preprocessing_task.context = [data_info_task]
            
            # Task 3: Data Preparation
            preparation_task = tasks_factory.prepare_data_task(
                agent=preprocessor,
                file_name=request.file_name,
                target_column=request.target_column,
                problem_type=request.problem_type,
                tools=tools,
                dataset_name=request.dataset_name
            )
            preparation_task.context = [preprocessing_task]
            
            # Task 4: Model Training
            training_task = tasks_factory.models_training_task(
                agent=ml_engineer,
                problem_type=request.problem_type,
                file_name=request.file_name,
                target_column=request.target_column,
                tools=tools,
                dataset_name=request.dataset_name
            )
            training_task.context = [preparation_task]
            
            # Task 5: Feature Importance
            feature_task = tasks_factory.feature_importance_analysis_task(
                agent=ml_engineer,
                file_name=request.file_name,
                target_column=request.target_column,
                problem_type=request.problem_type,
                tools=tools,
                dataset_name=request.dataset_name
            )
            feature_task.context = [training_task]
            
            crew = Crew(
                agents=[data_analyst, preprocessor, ml_engineer],
                tasks=[data_info_task, preprocessing_task, preparation_task, training_task, feature_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": result,
                "dataset_name": request.dataset_name,
                "pipeline_phases": ["data_analysis", "preprocessing", "preparation", "training", "feature_analysis"],
                "problem_type": request.problem_type,
                "target_column": request.target_column
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}
