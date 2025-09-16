from crewai import Task
from textwrap import dedent
from mcp.client.stdio import StdioServerParameters
import os

# ----------------------- MCP SERVER CONFIGURATION -----------------------
automl_tool = StdioServerParameters(
    command="python3",
    args=["./mcp_server/AutoML/servers/server.py"],
    env={"UV_PYTHON": "3.11", **os.environ},
)

# ------------------------ TASKS ------------------------
class Own_Task:
    def data_analysis_task(self, agent):
        return Task(
            description=dedent("""Perform comprehensive exploratory data analysis (EDA) on the dataset provided at {file_path}.
                The analysis should include statistical summaries, correlations, distributions, missing value analysis,
                and outlier detection. Generate visualizations using libraries such as matplotlib, seaborn, or plotly.
                Save all outputs in outputs/AutoML_Output/{dataset_name}_{task_type}"""),
            expected_output=dedent("""A report saved as outputs/AutoML_Output/{dataset_name}_{task_type}/eda_report.md,
                which includes:
                  a. Dataset statistics: shape, columns, dtypes, memory usage, missing values summary.
                  b. Correlation analysis with heatmaps and scatterplots.
                  c. Outlier detection results (boxplots, z-score/IQR methods).
                  d. Feature distributions (histograms, density plots).
                  e. Plain-language insights and recommendations for preprocessing.
                  f. interactive visualizations (plotly or seaborn)."""),
            output_file="outputs/AutoML_Output/{dataset_name}_{task_type}/eda_report.md",
            tools=[automl_tool],
            agent=agent,
        )

    def data_preprocessing_task(self, agent):
        return Task(
            description=dedent("""Perform thorough data preprocessing for the dataset at {file_path}.
                Include handling of missing values, duplicates, invalid entries, categorical encoding,
                feature scaling/normalization, and optional feature engineering.
                Save all preprocessed outputs and preprocessing logs in outputs/AutoML_Output/{dataset_name}_{task_type}/"""),
            expected_output=dedent("""A preprocessed dataset stored at outputs/AutoML_Output/{dataset_name}_{task_type}/preprocessed_dataset.csv
                along with a preprocessing report containing:
                  a. Detailed description of preprocessing steps.
                  b. Handling of missing data and duplicates.
                  c. Features engineered and rationale.
                  d. Validation results (e.g., no nulls, correct dtypes, clean encoding).
                  e. Summary of transformations applied to each column."""),
            output_file="outputs/AutoML_Output/{dataset_name}_{task_type}/preprocessed_dataset.csv",
            tools=[automl_tool],
            agent=agent,
        )

    def ml_model_task(self, agent):
        return Task(
            description=dedent("""Train and evaluate multiple machine learning models for {objective}.
                Use standard ML algorithms appropriate for the task (e.g., RandomForest, XGBoost, Logistic Regression, Linear Regression).
                Perform model evaluation, generate performance plots (ROC, confusion matrices, residuals),
                and summarize the results. Save all outputs in outputs/AutoML_Output/{dataset_name}_{task_type}/"""),
            expected_output=dedent("""A machine learning report saved as outputs/AutoML_Output/{dataset_name}_{task_type}/ml_models_report.md, including:
                  a. Training and evaluation metrics (accuracy, F1, precision, recall, RMSE, R2 depending on task).
                  b. Visualizations: residual plots, confusion matrices, ROC curves, precision-recall curves.
                  c. Comparison table of all trained models with key metrics.
                  d. Recommended best models with rationale.
                  e. Trained model files saved under outputs/AutoML_Output/{dataset_name}_{task_type}/models/"""),
            output_file="outputs/AutoML_Output/{dataset_name}_{task_type}/ml_models_report.md",
            tools=[automl_tool],
            agent=agent,
        )

    def model_evaluation_task(self, agent):
        return Task(
            description=dedent("""Conduct comprehensive evaluation of trained ML models for {objective}.
                Evaluate using appropriate metrics for classification or regression tasks.
                Include cross-validation results, confusion matrices, ROC curves, feature importance analysis,
                and comparative visualizations. Save evaluation outputs in outputs/AutoML_Output/{dataset_name}_{task_type}/"""),
            expected_output=dedent("""An evaluation report saved as outputs/AutoML_Output/{dataset_name}_{task_type}/model_evaluation.md, including:
                  a. Regression metrics (MSE, RMSE, R2) or Classification metrics (accuracy, F1, precision, recall, AUC).
                  b. Confusion matrix and ROC/PR curve visualizations.
                  c. Feature importance ranking (if applicable).
                  d. Model comparison table with trade-offs explained.
                  e. Insights and actionable recommendations."""),
            output_file="outputs/AutoML_Output/{dataset_name}_{task_type}/model_evaluation.md",
            tools=[automl_tool],
            agent=agent,
        )

    def hyperparameter_tuning_task(self, agent):
        return Task(
            description=dedent("""Optimize trained machine learning models for {objective} using hyperparameter tuning strategies:
                Grid Search, Random Search, or Bayesian Optimization.
                Log all trials and evaluate performance improvements. Retrain models with optimal hyperparameters
                and save results in outputs/AutoML_Output/{dataset_name}_{task_type}/"""),
            expected_output=dedent("""A hyperparameter tuning report saved as outputs/AutoML_Output/{dataset_name}_{task_type}/hyperparameter_tuning.md, including:
                  a. Search strategy used and trial logs.
                  b. Best hyperparameters discovered.
                  c. Comparison of default vs tuned models.
                  d. Performance improvement visualizations (bar charts, line plots).
                  e. Final retrained model saved at outputs/AutoML_Output/{dataset_name}_{task_type}/models/best_model.pkl"""),
            output_file="outputs/AutoML_Output/{dataset_name}_{task_type}/hyperparameter_tuning.md",
            tools=[automl_tool],
            agent=agent,
        )
