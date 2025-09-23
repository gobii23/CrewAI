from crewai import Task
from textwrap import dedent

class AutoMLTask:
    def automl_task(self, agent, dataset_name, file_path, output_path):
        return Task(
            description=dedent(f"""
                Execute a comprehensive end-to-end Machine Learning workflow using PyCaret MCP tools.
                You have access to powerful MCP tools for automated ML pipeline execution.
                
                IMPORTANT: All operations must run in SILENT MODE - no interactive displays or user prompts.
                
                MANDATORY EXECUTION SEQUENCE:
                
                **PHASE 1: Data Discovery & Validation**
                1. Use `list_datasets` tool to check if dataset '{dataset_name}' is already loaded
                2. If not loaded, use `load_dataset` tool with parameters:
                   - dataset_path: "{file_path}"
                   - dataset_name: "{dataset_name}"
                3. Analyze the loaded dataset structure, shape, and data types from the response
                
                **PHASE 2: Automated ML Pipeline Execution**
                
                OPTION A - Full Pipeline (Recommended):
                Use `full_ml_pipeline` tool with these parameters:
                - dataset_path: "{file_path}"
                - task_type: [Determine from context: "classification" or "regression"]
                - target_column: [Identify from dataset analysis or context clues]
                - output_dir: "{output_path}"
                - session_id: 123
                - train_size: 0.8
                - dataset_name: "{dataset_name}"
                
                OPTION B - Step-by-step Pipeline (If full pipeline fails):
                1. `setup_session` - Configure PyCaret ML environment with:
                   - dataset_name: "{dataset_name}"
                   - task_type: [classification/regression]
                   - target_column: [identified target]
                   - session_id: 123
                   - train_size: 0.8
                2. `compare_models` - Evaluate multiple algorithms with:
                   - session_key: [from setup_session response]
                   - n_select: 3
                3. `create_model` - Train the optimal model with:
                   - session_key: [from setup_session response]
                   - model_name: "rf" (or best from compare_models)
                4. `finalize_model` - Train on full dataset with:
                   - session_key: [from setup_session response]
                5. `save_artifacts` - Persist all outputs with:
                   - session_key: [from setup_session response]
                   - output_dir: "{output_path}"
                
                **PHASE 3: Results Analysis & Validation**
                1. Use `list_datasets` again to confirm dataset is properly loaded
                2. Verify all artifacts were saved in the output directory
                3. Document the complete workflow and results
                
                **TARGET COLUMN IDENTIFICATION STRATEGY:**
                - Look for columns named: 'target', 'label', 'class', 'outcome', 'y'
                - For classification: categorical columns with limited unique values
                - For regression: continuous numerical columns
                - If unclear, make an educated guess based on dataset context
                
                **TASK TYPE DETERMINATION:**
                - Classification: Target has discrete categories (< 20 unique values typically)
                - Regression: Target is continuous numerical data
                - Default to classification if uncertain
                
                **ERROR HANDLING:**
                - If any tool fails, try alternative approaches
                - Document any issues encountered and solutions applied
                - Ensure at least basic model training is completed
                
                **EXPECTED ARTIFACTS TO BE CREATED:**
                - Original dataset saved as CSV
                - Trained ML model pipeline (.pkl file)
                - Session configuration and metadata
                - Complete execution logs
                
                After successful execution, provide a comprehensive analysis of:
                - Data preprocessing steps applied
                - Model selection rationale and performance
                - Feature engineering recommendations
                - Deployment readiness assessment
                - Suggestions for model improvements
            """),
            
            expected_output=dedent("""
                **COMPREHENSIVE ML WORKFLOW EXECUTION REPORT**
                
                **1. DATASET ANALYSIS & PREPROCESSING SUMMARY**
                - Original dataset specifications (shape, columns, data types)
                - Data quality assessment (missing values, outliers, distributions)
                - Preprocessing transformations applied by PyCaret
                - Train/test split configuration and validation strategy
                - Feature engineering and encoding details
                
                **2. MODEL DEVELOPMENT & SELECTION RESULTS**
                - Task type identified: [Classification/Regression]
                - Target column selected and rationale
                - Model comparison results with performance metrics
                - Best model selected: [Model Name] with key parameters
                - Cross-validation scores and statistical significance
                - Hyperparameter optimization outcomes (if applied)
                
                **3. MODEL PERFORMANCE & EVALUATION**
                - Training performance metrics:
                  * Classification: Accuracy, Precision, Recall, F1-Score, AUC
                  * Regression: RMSE, MAE, R², MAPE
                - Validation set performance
                - Model interpretability insights (feature importance)
                - Overfitting/underfitting assessment
                
                **4. PIPELINE ARTIFACTS & REPRODUCIBILITY**
                - Complete list of saved artifacts with file paths:
                  * Original dataset: [path]
                  * Trained model pipeline: [path]
                  * Session configuration: [details]
                  * Performance metrics: [summary]
                - Session configuration details for reproducibility
                - Data integrity verification completed
                
                **5. DEPLOYMENT READINESS & RECOMMENDATIONS**
                - Production deployment checklist:
                  * Model serialization format confirmed
                  * Preprocessing pipeline compatibility verified
                  * Input data schema documented
                  * Performance benchmarks established
                
                **6. NEXT STEPS & IMPROVEMENTS**
                - Feature engineering opportunities identified
                - Model performance enhancement suggestions
                - Data collection recommendations for model improvement
                - Monitoring and retraining strategy recommendations
                - Business impact assessment and ROI projections
                
                **7. TECHNICAL IMPLEMENTATION NOTES**
                - PyCaret version and configuration used
                - Computational resources utilized
                - Execution time and performance metrics
                - Any issues encountered and resolutions applied
                - Alternative approaches considered
                
                **EXECUTIVE SUMMARY:**
                End-to-end AutoML pipeline successfully executed using PyCaret MCP tools. 
                Model [Model Name] achieved [Performance Metric] on [Dataset Name] with 
                full reproducibility artifacts saved to {output_path}. 
                Ready for production deployment with [X]% confidence level.
                
                **DEPLOYMENT COMMAND:**
                Model can be loaded for predictions using: `pickle.load(open('[model_path]', 'rb'))`
            """),
            
            agent=agent,
        )
