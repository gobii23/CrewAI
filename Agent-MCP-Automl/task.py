from crewai import Task
from textwrap import dedent


class AutoMLIndividualTasks:
    """
    AutoML Task definitions that accept MCP tools as parameters.
    This approach ensures MCP connections remain alive during task execution.
    """

    # ========================================
    # DATA INFORMATION AND LOADING TASKS
    # ========================================

    def information_about_data_task(self, agent, file_name: str, tools, dataset_name: str = None):
        """Get comprehensive dataset information using MCP tools"""
        information_tool = tools["information_about_data"]
        
        return Task(
            description=dedent(
                f"""
                Use the `information_about_data` tool to extract comprehensive dataset information.

                Inputs:
                - File Name: {file_name}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This tool provides detailed dataset statistics, metadata, and quality assessment.

                Your final answer MUST include:
                - Dataset shape (rows, columns)
                - Memory usage analysis
                - Data types for each column
                - Missing value counts and percentages
                - Basic statistical summaries
                """
            ),
            expected_output="A comprehensive dataset information report including shape, memory usage, data types, missing values analysis, and statistical summaries for all columns.",
            agent=agent,
            tools=[information_tool],
        )

    def reading_csv_task(self, agent, file_name: str, tools, dataset_name: str = None):
        """Load CSV dataset using MCP tools"""
        read_csv_tool = tools["read_csv_file"]
        
        return Task(
            description=dedent(
                f"""
                Use the `read_csv_file` tool to load and validate CSV dataset.

                Inputs:
                - File Name: {file_name}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This tool loads the CSV file and performs initial data validation.

                Your final answer MUST include:
                - File successfully loaded confirmation
                - Loading time and performance metrics
                - Any warnings or errors encountered
                - Initial data validation results
                """
            ),
            expected_output="A data loading report confirming successful file loading, performance metrics, validation results, and any warnings or errors encountered during the process.",
            agent=agent,
            tools=[read_csv_tool],
        )

    # ========================================
    # DATA VISUALIZATION TASKS
    # ========================================

    def visualize_correlation_numerical_task(self, agent, file_name: str, tools, dataset_name: str = None):
        """Generate numerical correlation matrix visualization"""
        correlation_num_tool = tools["visualize_correlation_numbers"]
        
        return Task(
            description=dedent(
                f"""
                Use the `visualize_correlation_numbers` tool to generate numerical correlation matrix visualization.

                Inputs:
                - File Name: {file_name}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This creates a heatmap visualization of numerical feature correlations.

                Your final answer MUST include:
                - Number of numerical features analyzed
                - Strong correlations identified (>0.7 or <-0.7)
                - Multicollinearity assessment
                - Feature relationship insights
                """
            ),
            expected_output="A numerical correlation analysis report with heatmap visualization, strong correlations identification, multicollinearity assessment, and feature relationship insights.",
            agent=agent,
            tools=[correlation_num_tool],
        )

    def visualize_correlation_categorical_task(self, agent, file_name: str, tools, dataset_name: str = None):
        """Generate categorical correlation matrix visualization"""
        correlation_cat_tool = tools["visualize_correlation_categories"]
        
        return Task(
            description=dedent(
                f"""
                Use the `visualize_correlation_categories` tool to generate categorical correlation matrix visualization.

                Inputs:
                - File Name: {file_name}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This creates association matrix visualization for categorical variables.

                Your final answer MUST include:
                - Number of categorical features analyzed
                - Association method used (Cramér's V, etc.)
                - Strong categorical associations identified
                - Redundancy assessment results
                """
            ),
            expected_output="A categorical correlation analysis report with association matrix visualization, strong associations identification, and categorical feature redundancy assessment.",
            agent=agent,
            tools=[correlation_cat_tool],
        )

    def visualize_correlation_final_task(self, agent, file_name: str, target_column: str, tools, dataset_name: str = None):
        """Generate final correlation matrix after preprocessing"""
        correlation_final_tool = tools["visualize_correlation_final"]
        
        return Task(
            description=dedent(
                f"""
                Use the `visualize_correlation_final` tool to generate final correlation matrix after preprocessing.

                Inputs:
                - File Name: {file_name}
                - Target Column: {target_column}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This creates post-preprocessing correlation analysis with target relationships.

                Your final answer MUST include:
                - Post-preprocessing correlation patterns
                - Features most correlated with target: {target_column}
                - Impact of preprocessing on correlations
                - Final feature insights and recommendations
                """
            ),
            expected_output="A final correlation analysis report showing post-preprocessing patterns, target variable correlations, preprocessing impact assessment, and feature selection recommendations.",
            agent=agent,
            tools=[correlation_final_tool],
        )

    def visualize_outliers_task(self, agent, file_name: str, tools, dataset_name: str = None):
        """Generate outlier detection visualizations"""
        outliers_tool = tools["visualize_outliers"]
        
        return Task(
            description=dedent(
                f"""
                Use the `visualize_outliers` tool to generate outlier detection visualizations.

                Inputs:
                - File Name: {file_name}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This creates comprehensive outlier analysis with box plots and statistical analysis.

                Your final answer MUST include:
                - Number of features analyzed for outliers
                - Outlier detection methods used
                - Features with highest outlier counts
                - Treatment recommendations
                """
            ),
            expected_output="An outlier detection analysis report with visualizations, outlier counts by feature, detection methods used, and treatment recommendations.",
            agent=agent,
            tools=[outliers_tool],
        )

    def visualize_outliers_final_task(self, agent, file_name: str, target_column: str, tools, dataset_name: str = None):
        """Generate outlier visualizations after preprocessing"""
        outliers_final_tool = tools["visualize_outliers_final"]
        
        return Task(
            description=dedent(
                f"""
                Use the `visualize_outliers_final` tool to generate final outlier visualizations after preprocessing.

                Inputs:
                - File Name: {file_name}
                - Target Column: {target_column}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This analyzes remaining outliers after preprocessing and treatment effectiveness.

                Your final answer MUST include:
                - Remaining outliers after treatment
                - Outlier treatment effectiveness
                - Impact on target variable: {target_column}
                - Final data quality assessment
                """
            ),
            expected_output="A final outlier analysis report showing post-treatment outlier status, treatment effectiveness, target relationship analysis, and data quality improvements.",
            agent=agent,
            tools=[outliers_final_tool],
        )

    # ========================================
    # DATA PREPROCESSING TASKS
    # ========================================

    def preprocessing_data_task(self, agent, file_name: str, target_column: str, tools, dataset_name: str = None):
        """Execute comprehensive data preprocessing"""
        preprocessing_tool = tools["preprocessing_data"]
        
        return Task(
            description=dedent(
                f"""
                Use the `preprocessing_data` tool to execute comprehensive data preprocessing.

                Inputs:
                - File Name: {file_name}
                - Target Column: {target_column}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This applies automated preprocessing pipeline including cleaning and transformations.

                Your final answer MUST include:
                - Preprocessing pipeline applied
                - Missing value handling methods used
                - Outlier treatment strategies applied
                - Data quality improvements achieved
                """
            ),
            expected_output="A comprehensive preprocessing report detailing pipeline steps, missing value treatment, outlier handling, data transformations applied, and quality improvements achieved.",
            agent=agent,
            tools=[preprocessing_tool],
        )

    def prepare_data_task(self, agent, file_name: str, target_column: str, problem_type: str, tools, dataset_name: str = None):
        """Prepare data for model training with encoding and scaling"""
        prepare_data_tool = tools["prepare_data"]
        
        return Task(
            description=dedent(
                f"""
                Use the `prepare_data` tool to prepare data for model training.

                Inputs:
                - File Name: {file_name}
                - Target Column: {target_column}
                - Problem Type: {problem_type}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This applies feature encoding, scaling, and creates train/test splits.

                Your final answer MUST include:
                - Problem type: {problem_type}
                - Categorical encoding methods used
                - Numerical scaling techniques applied
                - Train/validation/test splits created
                """
            ),
            expected_output="A data preparation report showing encoding methods, scaling techniques, feature engineering results, and train/test split configurations for the specified problem type.",
            agent=agent,
            tools=[prepare_data_tool],
        )

    # ========================================
    # MODEL TRAINING AND EVALUATION TASKS
    # ========================================

    def models_training_task(self, agent, problem_type: str, file_name: str, target_column: str, tools, dataset_name: str = None):
        """Train and evaluate multiple ML models"""
        models_tool = tools["select_and_evaluate_models"]
        
        return Task(
            description=dedent(
                f"""
                Use the `select_and_evaluate_models` tool to train and evaluate multiple machine learning models.

                Inputs:
                - Problem Type: {problem_type}
                - File Name: {file_name}
                - Target Column: {target_column}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This trains multiple algorithms and compares their performance.

                Your final answer MUST include:
                - Problem type: {problem_type}
                - Algorithms evaluated
                - Performance metrics for each model
                - Best performing model identified
                """
            ),
            expected_output="A comprehensive model training report showing algorithms evaluated, performance metrics comparison, cross-validation results, and best model identification with ranking.",
            agent=agent,
            tools=[models_tool],
        )

    def visualize_accuracy_matrix_task(self, agent, file_name: str, target_column: str, problem_type: str, tools, dataset_name: str = None):
        """Generate accuracy/confusion matrix visualizations"""
        accuracy_matrix_tool = tools["visualize_accuracy_matrix"]
        
        return Task(
            description=dedent(
                f"""
                Use the `visualize_accuracy_matrix` tool to generate accuracy matrix and performance visualizations.

                Inputs:
                - File Name: {file_name}
                - Target Column: {target_column}
                - Problem Type: {problem_type}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This creates performance visualizations like confusion matrices or residual plots.

                Your final answer MUST include:
                - Problem type: {problem_type}
                - Visualization types generated
                - Performance metrics displayed
                - Error analysis insights
                """
            ),
            expected_output="A performance visualization report with accuracy matrices, confusion matrices or residual plots, performance metrics analysis, and error pattern identification.",
            agent=agent,
            tools=[accuracy_matrix_tool],
        )

    # ========================================
    # HYPERPARAMETER OPTIMIZATION TASK
    # ========================================

    def best_model_hyperparameter_task(
        self,
        agent,
        model_name: str,
        file_name: str,
        target_column: str,
        problem_type: str,
        tools,
        n_trials: int = 100,
        scoring: str = "auto",
        random_state: int = 42,
        dataset_name: str = None,
    ):
        """Optimize hyperparameters for the best model"""
        hyperparameter_tool = tools["best_model_hyperparameter"]
        
        return Task(
            description=dedent(
                f"""
                Use the `best_model_hyperparameter` tool to optimize hyperparameters for the best performing model.

                Inputs:
                - Model Name: {model_name}
                - File Name: {file_name}
                - Target Column: {target_column}
                - Problem Type: {problem_type}
                - Number of Trials: {n_trials}
                - Scoring Metric: {scoring}
                - Random State: {random_state}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This executes hyperparameter optimization using advanced search algorithms.

                Your final answer MUST include:
                - Model optimized: {model_name}
                - Best hyperparameters found
                - Performance improvement achieved
                - Optimization process analysis
                """
            ),
            expected_output="A hyperparameter optimization report showing the model optimized, best parameters found, performance improvements, parameter importance analysis, and optimization convergence details.",
            agent=agent,
            tools=[hyperparameter_tool],
        )

    # ========================================
    # PREDICTION AND TESTING TASKS
    # ========================================

    def test_external_data_task(
        self,
        agent,
        main_file_name: str,
        target_column: str,
        problem_type: str,
        test_file_name: str,
        tools,
        dataset_name: str = None,
    ):
        """Test model on external dataset"""
        test_external_tool = tools["test_external_data"]
        
        return Task(
            description=dedent(
                f"""
                Use the `test_external_data` tool to test the trained model on external data.

                Inputs:
                - Main File Name: {main_file_name}
                - Target Column: {target_column}
                - Problem Type: {problem_type}
                - Test File Name: {test_file_name}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This applies the trained model to external test data for generalization assessment.

                Your final answer MUST include:
                - External test data: {test_file_name}
                - Predictions generated successfully
                - Performance on external data
                - Generalization assessment results
                """
            ),
            expected_output="An external data testing report showing predictions on test data, performance metrics on unseen data, generalization capability assessment, and model robustness evaluation.",
            agent=agent,
            tools=[test_external_tool],
        )

    def predict_value_task(
        self,
        agent,
        model_name: str,
        file_name: str,
        target_column: str,
        problem_type: str,
        input_data: str,
        tools,
        n_trials: int = 100,
        scoring: str = "auto",
        random_state: int = 42,
        dataset_name: str = None,
    ):
        """Generate predictions for new input data"""
        predict_tool = tools["predict_value"]
        
        return Task(
            description=dedent(
                f"""
                Use the `predict_value` tool to generate predictions for new input data.

                Inputs:
                - Model Name: {model_name}
                - File Name: {file_name}
                - Target Column: {target_column}
                - Problem Type: {problem_type}
                - Input Data: {input_data}
                - Number of Trials: {n_trials}
                - Scoring Metric: {scoring}
                - Random State: {random_state}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This processes input data and generates predictions with confidence scores.

                Your final answer MUST include:
                - Model used: {model_name}
                - Predicted value(s)
                - Confidence/probability scores
                - Feature contribution analysis
                """
            ),
            expected_output="A prediction report showing the model used, predicted values, confidence scores, prediction intervals, feature importance for this prediction, and reliability assessment.",
            agent=agent,
            tools=[predict_tool],
        )

    # ========================================
    # FEATURE ANALYSIS TASK
    # ========================================

    def feature_importance_analysis_task(
        self,
        agent,
        file_name: str,
        target_column: str,
        problem_type: str,
        tools,
        dataset_name: str = None,
    ):
        """Analyze feature importance using XGBoost"""
        feature_importance_tool = tools["feature_importance_analysis"]
        
        return Task(
            description=dedent(
                f"""
                Use the `feature_importance_analysis` tool to analyze feature importance using XGBoost.

                Inputs:
                - File Name: {file_name}
                - Target Column: {target_column}
                - Problem Type: {problem_type}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                This generates comprehensive feature importance analysis and rankings.

                Your final answer MUST include:
                - Target variable: {target_column}
                - Problem type: {problem_type}
                - Top 10 most important features
                - Feature importance scores and percentages
                - Feature selection recommendations
                """
            ),
            expected_output="A feature importance analysis report showing top important features, importance scores, cumulative importance analysis, feature selection recommendations, and business insights.",
            agent=agent,
            tools=[feature_importance_tool],
        )

    # ========================================
    # COMPLETE PIPELINE TASK
    # ========================================

    def complete_automl_pipeline_task(
        self,
        agent,
        file_name: str,
        target_column: str,
        problem_type: str,
        tools,
        dataset_name: str = None,
    ):
        """Execute complete AutoML pipeline end-to-end using multiple MCP tools"""
        # Get all required tools
        info_tool = tools["information_about_data"]
        read_tool = tools["read_csv_file"]
        preprocess_tool = tools["preprocessing_data"]
        prepare_tool = tools["prepare_data"]
        models_tool = tools["select_and_evaluate_models"]
        feature_tool = tools["feature_importance_analysis"]
        
        return Task(
            description=dedent(
                f"""
                Execute a complete AutoML pipeline using multiple MCP tools in sequence.

                Pipeline Steps:
                1. Use `information_about_data` for dataset analysis
                2. Use `read_csv_file` to load the data
                3. Use `preprocessing_data` for data cleaning
                4. Use `prepare_data` for feature engineering
                5. Use `select_and_evaluate_models` for model training
                6. Use `feature_importance_analysis` for interpretability

                Inputs:
                - File Name: {file_name}
                - Target Column: {target_column}
                - Problem Type: {problem_type}
                - Dataset Name: {dataset_name if dataset_name else "Auto-generated"}

                Execute each step systematically and provide comprehensive results.

                Your final answer MUST include:
                - Dataset analysis summary
                - Data preprocessing results
                - Best model identified
                - Feature importance insights
                - Complete pipeline execution status
                """
            ),
            expected_output="A comprehensive AutoML pipeline report covering dataset analysis, preprocessing results, model training outcomes, feature importance analysis, and complete execution summary with performance metrics.",
            agent=agent,
            tools=[info_tool, read_tool, preprocess_tool, prepare_tool, models_tool, feature_tool],
        )