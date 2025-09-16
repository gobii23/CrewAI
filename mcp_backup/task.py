from crewai import Task
from textwrap import dedent
from mcp.client.stdio import StdioServerParameters
from crewai_tools import MCPServerAdapter
import os

# MCP Server Configuration
automl_tool = StdioServerParameters(
    command="python3",
    args=["./mcp_server/AutoML/server.py"],
    env={"UV_PYTHON": "3.11", **os.environ},
)

class AutoMLTasks:
    def __init__(self):
        self.output_base = "outputs/AutoML_Output"
    
    def create_output_path(self, dataset_name: str, task_type: str) -> str:
        return f"{self.output_base}/{dataset_name}_{task_type}"
    
    def data_analysis_task(self, agent, file_path: str, dataset_name: str):
        output_path = self.create_output_path(dataset_name, "EDA")
        return Task(
            description=dedent(f"""
            Perform comprehensive exploratory data analysis on the dataset at {file_path}.
            Use the available MCP tools to analyze the data with these parameters:
            - file_path: "{file_path}"
            - dataset_name: "{dataset_name}"
            - output_path: "{output_path}"
            Your analysis must include:
            1. Dataset overview (shape, columns, data types, memory usage)
            2. Missing value analysis with visualization
            3. Statistical summaries for numerical and categorical features
            4. Correlation analysis with heatmaps
            5. Distribution analysis with appropriate plots
            6. Outlier detection using multiple methods
            7. Feature relationship exploration
            8. Data quality assessment
            Save all outputs to: {output_path}/
            Generate interactive visualizations where possible.
            """),
            expected_output=dedent(f"""
            Complete EDA report saved as {output_path}/eda_report.md containing:
            1. Executive summary with key findings
            2. Dataset statistics and structure analysis  
            3. Missing value patterns and recommendations
            4. Correlation insights and multicollinearity warnings
            5. Distribution analysis with skewness/kurtosis metrics
            6. Outlier detection results with impact assessment
            7. Feature importance preliminary analysis
            8. Preprocessing recommendations based on findings
            9. All visualizations saved as PNG/HTML files
            """),
            output_file=f"{output_path}/eda_report.md",
            tools=[automl_tool],
            agent=agent,
        )
    
    def preprocessing_task(self, agent, file_path: str, dataset_name: str):
        output_path = self.create_output_path(dataset_name, "preprocessing")
        return Task(
            description=dedent(f"""
            Execute comprehensive data preprocessing on {file_path} based on EDA findings.
            Required preprocessing steps:
            1. Handle missing values (imputation, removal, or flagging)
            2. Remove or flag duplicate records
            3. Encode categorical variables appropriately
            4. Scale/normalize numerical features as needed
            5. Handle outliers based on domain knowledge
            6. Create derived features if beneficial
            7. Ensure data type consistency
            8. Split data into train/validation/test sets
            Document all transformations for reproducibility.
            Save outputs to: {output_path}/
            """),
            expected_output=dedent(f"""
            Preprocessed dataset and documentation saved to {output_path}/:
            1. preprocessed_train.csv, preprocessed_val.csv, preprocessed_test.csv
            2. preprocessing_report.md with detailed transformation log
            4. transformation_pipeline.pkl for future use
            5. data_dictionary.csv mapping original to transformed features
            6. preprocessing_summary.json with statistics before/after
            7. validation_checks.md confirming data quality
            """),
            output_file=f"{output_path}/preprocessing_report.md",
            tools=[automl_tool],
            agent=agent,
        )
    
    def model_training_task(self, agent, dataset_name: str, objective: str):
        output_path = self.create_output_path(dataset_name, "training")
        return Task(
            description=dedent(f"""
            Train multiple ML models for {objective} using preprocessed data.
            Model training requirements:
            1. Select appropriate algorithms based on problem type and data size
            2. Implement proper cross-validation strategy
            3. Train baseline models (3-5 different algorithms)
            4. Generate performance metrics appropriate for {objective}
            5. Create model comparison visualizations
            6. Save trained models and predictions
            7. Document model selection rationale
            Focus on interpretability alongside performance.
            Save all outputs to: {output_path}/
            """),
            expected_output=dedent(f"""
            Complete model training results saved to {output_path}/:
            1. model_training_report.md with detailed analysis
            2. models/ directory with trained model files (.pkl)
            3. predictions/ directory with train/val/test predictions
            4. metrics_comparison.csv with all model performance metrics
            5. model_performance_plots/ with ROC, confusion matrix, residuals
            6. feature_importance_analysis.md
            7. model_selection_recommendation.md
            """),
            output_file=f"{output_path}/model_training_report.md",
            tools=[automl_tool],
            agent=agent,
        )
    
    def hyperparameter_optimization_task(self, agent, dataset_name: str, objective: str):
        output_path = self.create_output_path(dataset_name, "optimization")
        return Task(
            description=dedent(f"""
            Optimize hyperparameters for best performing models from training phase.
            Optimization requirements:
            1. Select top 2-3 models for hyperparameter tuning
            2. Define appropriate parameter spaces for each model
            3. Use efficient search strategies (Bayesian, Random, or Grid)
            4. Implement proper validation to avoid overfitting
            5. Compare optimized vs baseline performance
            6. Generate optimization history and convergence plots
            7. Final model retraining with optimal parameters
            Save all optimization results to: {output_path}/
            """),
            expected_output=dedent(f"""
            Hyperparameter optimization results saved to {output_path}/:
            1. optimization_report.md with detailed analysis
            2.optimized_models/ directory with tuned models
            3. optimization_history.csv with all trial results
            4. parameter_importance_plots/ 
            5. convergence_analysis.md
            6. final_model_comparison.md (baseline vs optimized)
            7. best_model.pkl (production-ready model)
            8. model_deployment_guide.md
            """),
            output_file=f"{output_path}/optimization_report.md",
            tools=[automl_tool],
            agent=agent,
        )
    
    def final_evaluation_task(self, agent, dataset_name: str, objective: str):
        output_path = self.create_output_path(dataset_name, "final_evaluation")
        return Task(
            description=dedent(f"""
            Conduct comprehensive final evaluation of the optimized model.
            Evaluation requirements:
            1. Test set performance evaluation (never seen during training/optimization)
            2. Model robustness analysis (different data subsets)
            3. Feature importance and SHAP analysis for interpretability
            4. Model limitations and failure case analysis
            5. Business impact assessment
            6. Deployment readiness checklist
            7. Model monitoring recommendations
            Provide production deployment recommendations.
            Save all outputs to: {output_path}/
            """),
            expected_output=dedent(f"""
            Final evaluation package saved to {output_path}/:
            1. final_evaluation_report.md (executive summary)
            2. test_performance_analysis.md
            3. model_interpretability_report.md with SHAP analysis
            4. robustness_analysis.md
            5. failure_case_analysis.md  
            6. business_impact_assessment.md
            7. deployment_checklist.md
            8. monitoring_recommendations.md
            9. model_card.md (standardized documentation)
            """),
            output_file=f"{output_path}/final_evaluation_report.md",
            tools=[automl_tool],
            agent=agent,
        )
