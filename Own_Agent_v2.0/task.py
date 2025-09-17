from crewai import Task
from textwrap import dedent
from mcp import StdioServerParameters
import os


class AutoMLTasks:
    def __init__(self):
        self.output_base = "outputs/AutoML_Output"

        # Updated MCP Server Configuration for the AutoML server we created
        self.server_params = StdioServerParameters(
            command="docker",
            args=["exec", "-i", "automl-mcp-server", "python", "automl_server.py"],
            env={"PYTHONPATH": ".", **os.environ},
        )

    def create_output_path(self, dataset_name: str, task_type: str) -> str:
        return f"{self.output_base}/{dataset_name}_{task_type}"

    def data_analysis_task(self, agent, file_path: str, dataset_name: str):
        output_path = self.create_output_path(dataset_name, "EDA")
        return Task(
            description=dedent(
                f"""
            Perform comprehensive exploratory data analysis on the dataset at {file_path}.
            Use the AutoML MCP tools in this sequence:
            1. First, load the dataset:
               - Use load_dataset tool with dataset_path="{file_path}" and dataset_name="{dataset_name}"
            2. Then analyze the data:
               - Use data_analysis tool with dataset_name="{dataset_name}"
            The analysis should automatically include:
            - Dataset overview (shape, columns, data types, memory usage)
            - Missing value analysis with visualizations
            - Statistical summaries for numerical and categorical features
            - Correlation analysis with heatmaps
            - Distribution analysis with appropriate plots
            - Outlier detection using multiple methods
            - Feature relationship exploration
            - Data quality assessment
            Extract key insights from the analysis results and provide actionable recommendations
            for preprocessing steps and save the images. Focus on data quality issues, feature relationships, and
            patterns that will inform downstream ML tasks.
            Save your analysis report and images to: {output_path}/
            """
            ),
            expected_output=dedent(
                f"""
            Complete EDA report saved as {output_path}/eda_report.md containing:
            1. Executive summary with key findings and data quality assessment
            2. Dataset statistics and structure analysis with recommendations
            3. Missing value patterns and suggested handling strategies
            4. Correlation insights and multicollinearity warnings
            5. Distribution analysis with skewness/kurtosis metrics and normalization needs
            6. Outlier detection results with impact assessment and handling recommendations
            7. Feature importance preliminary analysis and feature engineering suggestions
            8. Preprocessing recommendations based on findings
            9. Data visualization summaries and insights
            10. Next steps for preprocessing phase
            """
            ),
            output_file=f"{output_path}/eda_report.md",
            agent=agent,
        )

    def preprocessing_task(self, agent, file_path: str, dataset_name: str):
        output_path = self.create_output_path(dataset_name, "preprocessing")
        return Task(
            description=dedent(
                f"""
            Execute comprehensive data preprocessing based on EDA findings.
            Use the AutoML MCP tools in sequence:
            1. Ensure dataset is loaded (if not already):
               - Use load_dataset tool if needed with dataset_path="{file_path}" and dataset_name="{dataset_name}"
            2. Apply preprocessing pipeline:
               - Use preprocess_data tool with dataset_name="{dataset_name}" and appropriate target_column
               - Configure preprocessing_config as JSON string based on EDA findings, for example:
                 {{"handle_missing": "mean", "encode_categorical": "onehot", "scale_features": "standard", "handle_outliers": "cap", "test_size": 0.2}}
            Base your preprocessing decisions on the EDA results from the previous task.
            Consider the following based on data characteristics:
            - Missing value strategy (mean, median, KNN, or drop)
            - Categorical encoding method (onehot or label encoding)
            - Scaling approach (standard, minmax, or none)
            - Outlier handling (remove, cap, or none)
            - Train/test split ratio
            Document all preprocessing decisions and their rationale.
            Save your preprocessing report to: {output_path}/
            """
            ),
            expected_output=dedent(
                f"""
            Preprocessed dataset documentation saved to {output_path}/preprocessing_report.md:
            1. Preprocessing steps executed with detailed rationale
            2. Before/after data statistics comparison
            3. Train/validation/test split information
            4. Feature transformations applied (encoding, scaling, etc.)
            5. Missing value handling strategy and results
            6. Outlier treatment approach and impact
            7. Data quality validation checks
            8. Feature engineering recommendations for future iterations
            9. Preprocessed data summary (shape, features, target distribution)
            10. Ready-to-use dataset confirmation for model training
            """
            ),
            output_file=f"{output_path}/preprocessing_report.md",
            agent=agent,
        )

    def model_training_task(self, agent, dataset_name: str, objective: str):
        output_path = self.create_output_path(dataset_name, "training")
        return Task(
            description=dedent(
                f"""
            Train and evaluate multiple baseline machine learning models.
            Use the AutoML MCP tools:
            1. Train baseline models:
               - Use train_baseline_models tool with dataset_name="{dataset_name}"
               - Configure model_config as needed, e.g.: {{"cv_folds": 5, "random_state": 42}}
            The training will automatically:
            - Select appropriate algorithms based on problem type (classification/regression)
            - Train multiple baseline models (Random Forest, XGBoost, LightGBM, etc.)
            - Perform cross-validation for robust performance estimation
            - Generate performance metrics appropriate for {objective}
            - Create model comparison visualizations
            - Save trained models and predictions
            Analyze the results and provide insights about:
            - Which algorithms perform best for this dataset and why
            - Performance differences across models
            - Signs of overfitting or underfitting
            - Model interpretability vs performance trade-offs
            - Recommendations for hyperparameter optimization phase
            Save your analysis to: {output_path}/
            """
            ),
            expected_output=dedent(
                f"""
            Complete model training analysis saved to {output_path}/model_training_report.md:
            1. Model performance comparison with cross-validation results
            2. Best performing algorithms identification and analysis
            3. Performance metrics breakdown (accuracy, F1, precision, recall for classification OR R², RMSE, MAE for regression)
            4. Model strengths and weaknesses analysis
            5. Overfitting/underfitting assessment across models
            6. Feature importance insights from tree-based models
            7. Training time and computational complexity comparison
            8. Model selection recommendations for optimization phase
            9. Business impact interpretation of model performance
            10. Next steps for hyperparameter optimization
            """
            ),
            output_file=f"{output_path}/model_training_report.md",
            agent=agent,
        )

    def hyperparameter_optimization_task(
        self, agent, dataset_name: str, objective: str
    ):
        output_path = self.create_output_path(dataset_name, "optimization")
        return Task(
            description=dedent(
                f"""
            Optimize hyperparameters for the best performing models from training phase.
            Use the AutoML MCP tools:
            1. Run hyperparameter optimization:
               - Use optimize_hyperparameters tool with dataset_name="{dataset_name}"
               - Set models_to_optimize="" to auto-select top 3 models, or specify manually
               - Configure optimization_config, e.g.: {{"n_trials": 100, "cv_folds": 5}}
            The optimization will automatically:
            - Select top performing models from baseline training
            - Define appropriate parameter spaces for each model type
            - Use Bayesian optimization (Optuna) for efficient search
            - Implement proper cross-validation to avoid overfitting
            - Compare optimized vs baseline performance
            - Generate optimization history and convergence analysis
            - Retrain final models with optimal parameters
            Analyze the optimization results and provide insights about:
            - Performance improvements achieved through optimization
            - Parameter sensitivity and importance
            - Optimization convergence and search efficiency  
            - Final model selection rationale
            - Diminishing returns analysis
            Save your analysis to: {output_path}/
            """
            ),
            expected_output=dedent(
                f"""
            Hyperparameter optimization analysis saved to {output_path}/optimization_report.md:
            1. Optimization results summary with performance improvements
            2. Best parameters found for each optimized model
            3. Baseline vs optimized performance comparison
            4. Parameter importance and sensitivity analysis
            5. Optimization convergence analysis and trial efficiency
            6. Computational cost vs performance gain assessment
            7. Final model selection and ranking with detailed rationale
            8. Statistical significance of improvements
            9. Model stability and robustness evaluation
            10. Production deployment readiness assessment
            """
            ),
            output_file=f"{output_path}/optimization_report.md",
            agent=agent,
        )

    def final_evaluation_task(self, agent, dataset_name: str, objective: str):
        output_path = self.create_output_path(dataset_name, "final_evaluation")
        return Task(
            description=dedent(
                f"""
            Conduct comprehensive final evaluation of the optimized model.
            Use the AutoML MCP tools:
            1. Perform final evaluation:
               - Use final_evaluation tool with dataset_name="{dataset_name}"
               - Specify model_name="" to auto-select best model or choose specific model
            2. Generate comprehensive report:
               - Use generate_report tool with dataset_name="{dataset_name}" and report_type="full"
            3. Save the final model:
               - Use save_model tool with dataset_name="{dataset_name}", model_type="final"
            The evaluation will automatically include:
            - Test set performance evaluation (holdout data never seen during training/optimization)
            - Model robustness analysis across different data subsets
            - Feature importance and SHAP analysis for interpretability
            - Model limitations and failure case analysis
            - Business impact assessment and deployment readiness
            Provide comprehensive analysis covering:
            - Model performance on unseen data
            - Interpretability insights and feature contributions
            - Model reliability and robustness
            - Deployment considerations and monitoring recommendations
            - Business value proposition and ROI expectations
            Save your final analysis to: {output_path}/
            """
            ),
            expected_output=dedent(
                f"""
            Final evaluation package saved to {output_path}/final_evaluation_report.md:
            1. Executive summary with key findings and model recommendation
            2. Test set performance analysis with confidence intervals
            3. Model interpretability report with SHAP analysis and feature importance
            4. Robustness analysis across data subsets and edge cases
            5. Failure case analysis with recommendations for improvement
            6. Business impact assessment with ROI projections
            7. Model limitations and assumptions documentation
            8. Deployment checklist with infrastructure requirements
            9. Monitoring recommendations for production environment
            10. Model maintenance and retraining strategy
            11. Risk assessment and mitigation strategies
            12. Final model artifacts and documentation package
            """
            ),
            output_file=f"{output_path}/final_evaluation_report.md",
            agent=agent,
        )

    def cleanup_task(self, agent, dataset_name: str):
        """Optional cleanup task to free memory after pipeline completion."""
        return Task(
            description=dedent(
                f"""
            Clean up AutoML pipeline resources and provide summary.
            Use the AutoML MCP tools:
            1. List final datasets and results:
               - Use list_datasets tool with show_details="true"
            2. Optional cleanup (if needed):
               - Use cleanup_data tool with dataset_name="{dataset_name}" and cleanup_type="all"
               - Only perform cleanup if explicitly requested or memory is constrained
            Provide a final summary of the entire AutoML pipeline execution,
            including all artifacts created and their locations.
            """
            ),
            expected_output=dedent(
                f"""
            Pipeline completion summary:
            1. All datasets processed and stored status
            2. Models trained and optimized summary
            3. Output artifacts inventory with file locations
            4. Memory usage and cleanup status
            5. Pipeline execution time and resource utilization
            6. Success metrics and completion confirmation
            """
            ),
            agent=agent,
        )
