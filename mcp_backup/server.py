import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from datetime import datetime
from fastmcp import FastMCP
import pycaret.classification as pycl
import pycaret.regression as pycr


mcp = FastMCP("AutoML Server")

# Global storage
SESSIONS: Dict[str, Dict] = {}
DATASETS: Dict[str, pd.DataFrame] = {}

TaskType = Literal["classification", "regression"]


@mcp.tool
def load_dataset(
    dataset_path: str, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Load a dataset from file path into memory."""
    path = Path(dataset_path)
    if dataset_name is None:
        dataset_name = path.stem

    # Load based on file extension
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(dataset_path)
    else:
        df = pd.read_json(dataset_path)

    DATASETS[dataset_name] = df

    # Fix serialization issue with dtypes
    dtypes_serializable = {}
    for col, dtype in df.dtypes.items():
        dtypes_serializable[col] = str(dtype)

    return {
        "status": "success",
        "dataset_name": dataset_name,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": dtypes_serializable,
    }




@mcp.tool
def setup_session(dataset_name: str, task_type: TaskType, target_column: str, session_id: int = 123, train_size: float = 0.8) -> Dict[str, Any]:
    """Setup PyCaret ML session."""
    df = DATASETS[dataset_name]

    if task_type == "classification":
        session = pycl.setup(
            data=df,
            target=target_column,
            session_id=session_id,
            train_size=train_size,
        )
    else:
        session = pycr.setup(
            data=df,
            target=target_column,
            session_id=session_id,
            train_size=train_size,
        )

    session_key = f"{dataset_name}_{task_type}_{session_id}"
    SESSIONS[session_key] = {
        "dataset_name": dataset_name,
        "task_type": task_type,
        "target_column": target_column,
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
    }

    return {"status": "success", "session_key": session_key, "task_type": task_type}


@mcp.tool
def compare_models(session_key: str, n_select: int = 3) -> Dict[str, Any]:
    """Compare multiple ML models."""
    session_info = SESSIONS[session_key]
    task_type = session_info["task_type"]

    if task_type == "classification":
        models = pycl.compare_models(n_select=n_select)
    else:
        models = pycr.compare_models(n_select=n_select)

    SESSIONS[session_key]["models"] = models

    model_names = [
        str(type(model).__name__)
        for model in (models if isinstance(models, list) else [models])
    ]

    return {"status": "success", "models": model_names, "session_key": session_key}


@mcp.tool
def create_model(session_key: str, model_name: str = "rf") -> Dict[str, Any]:
    """Create and train a model."""
    session_info = SESSIONS[session_key]
    task_type = session_info["task_type"]

    if task_type == "classification":
        model = pycl.create_model(model_name, verbose=False)
    else:
        model = pycr.create_model(model_name, verbose=False)

    SESSIONS[session_key]["best_model"] = model

    return {"status": "success", "model_type": str(type(model).__name__)}


@mcp.tool
def finalize_model(session_key: str) -> Dict[str, Any]:
    """Finalize model by training on entire dataset."""
    session_info = SESSIONS[session_key]
    task_type = session_info["task_type"]
    model = session_info["best_model"]

    if task_type == "classification":
        final_model = pycl.finalize_model(model, verbose=False)
    else:
        final_model = pycr.finalize_model(model, verbose=False)

    SESSIONS[session_key]["final_model"] = final_model

    return {"status": "success", "session_key": session_key}


@mcp.tool
def save_artifacts(session_key: str, output_dir: str) -> Dict[str, Any]:
    """Save model and artifacts to disk."""
    session_info = SESSIONS[session_key]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Save dataset
    dataset_name = session_info["dataset_name"]
    if dataset_name in DATASETS:
        dataset_file = output_path / f"{dataset_name}.csv"
        DATASETS[dataset_name].to_csv(dataset_file, index=False)
        saved_files.append(str(dataset_file))

    # Save model
    if "final_model" in session_info:
        model_file = output_path / f"{session_key}_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(session_info["final_model"], f)
        saved_files.append(str(model_file))

    return {
        "status": "success",
        "saved_files": saved_files,
        "output_dir": str(output_path),
    }


@mcp.tool
def full_ml_pipeline(dataset_path: str, task_type: TaskType, target_column: str, output_dir: str, dataset_name: Optional[str] = None, session_id: int = 123, train_size: float = 0.8) -> Dict[str, Any]:
    """Execute complete end-to-end ML pipeline."""
    # Step 1: Load data
    load_result = load_dataset(dataset_path, dataset_name)
    dataset_name = load_result["dataset_name"]

    # Step 2: Setup session
    session_result = setup_session(
        dataset_name, task_type, target_column, session_id, train_size
    )
    session_key = session_result["session_key"]

    # Step 3: Compare models
    compare_result = compare_models(session_key)

    # Step 4: Create best model
    create_result = create_model(session_key)

    # Step 5: Finalize model
    finalize_result = finalize_model(session_key)

    # Step 6: Save artifacts
    save_result = save_artifacts(session_key, output_dir)

    return {
        "status": "success",
        "session_key": session_key,
        "dataset_shape": load_result["shape"],
        "model_type": create_result["model_type"],
        "saved_files": save_result["saved_files"],
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    mcp.run()