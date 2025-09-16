# AutoML MCP Server

A comprehensive Model Context Protocol (MCP) server that provides automated machine learning capabilities including data analysis, preprocessing, model training, hyperparameter optimization, and evaluation.

## Purpose

This MCP server provides a complete AutoML pipeline for AI assistants to perform end-to-end machine learning tasks with minimal manual intervention. It handles the entire ML workflow from data loading to model deployment preparation.

## Features

### Current Implementation

- **`load_dataset`** - Load CSV/JSON datasets and store for analysis
- **`data_analysis`** - Comprehensive data exploration with visualizations and quality assessment
- **`preprocess_data`** - Automated data preprocessing including missing value imputation, encoding, scaling, and train-test split
- **`train_baseline_models`** - Train and compare multiple baseline models (RF, XGBoost, LightGBM, etc.)
- **`optimize_hyperparameters`** - Advanced hyperparameter optimization using Optuna with Bayesian optimization
- **`final_evaluation`** - Complete model evaluation with interpretability analysis using SHAP
- **`save_model`** - Save trained models to disk using joblib
- **`generate_report`** - Create comprehensive AutoML reports in markdown format
- **`list_datasets`** - View all loaded datasets and their pipeline status
- **`cleanup_data`** - Clean up memory by removing stored data, models, and results

## Prerequisites

- Docker Desktop with MCP Toolkit enabled
- Docker MCP CLI plugin (`docker mcp` command)
- Sufficient system memory for data processing and model training
- Optional: GPU support for faster training with XGBoost/LightGBM

## Installation

See the step-by-step instructions provided with the files.

## Usage Examples

In Claude Desktop, you can ask:

### Data Loading and Analysis
- "Load the dataset from 'data.csv' and call it 'sales_data'"
- "Analyze the sales_data dataset and show me the data quality issues"
- "What are the main characteristics of my dataset?"

### Data Preprocessing
- "Preprocess the sales_data with 'revenue' as target column using standard scaling"
- "Handle missing values with KNN imputation and use one-hot encoding for categorical variables"

### Model Training
- "Train baseline models on the preprocessed sales_data"
- "Which model performs best on my dataset?"

### Hyperparameter Optimization
- "Optimize hyperparameters for Random Forest and XGBoost models with 200 trials"
- "Fine-tune the top 3 performing models"

### Final Evaluation
- "Perform final evaluation and generate interpretability analysis"
- "Show me SHAP analysis for the best model"

### Reporting and Management
- "Generate a full AutoML report for sales_data"
- "List all my datasets and their pipeline status"
- "Save the final model as 'production_model.joblib'"
- "Clean up all data and models to free memory"

## Architecture

```
Claude Desktop → MCP Gateway → AutoML MCP Server → ML Libraries
                                     ↓
                              pandas, scikit-learn,
                              XGBoost, LightGBM, SHAP,
                              Optuna, matplotlib, plotly
```

## ML Pipeline Workflow

1. **Data Loading**: Load datasets from CSV/JSON files
2. **Data Analysis**: Comprehensive EDA with visualizations
3. **Preprocessing**: Automated data cleaning and preparation
4. **Baseline Training**: Train multiple algorithms and compare
5. **Optimization**: Hyperparameter tuning with Bayesian optimization
6. **Final Evaluation**: Complete assessment with interpretability
7. **Deployment**: Model saving and comprehensive reporting

## Supported Algorithms

### Classification
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM  
- Gradient Boosting
- Support Vector Machine
- Naive Bayes

### Regression
- Linear Regression
- Ridge Regression
- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- Support Vector Regression

## Key Features

### Advanced Data Analysis
- Missing value analysis with heatmaps
- Correlation analysis
- Distribution analysis
- Outlier detection using IQR method
- Data quality assessment

### Intelligent Preprocessing
- Multiple imputation strategies (mean, median, KNN)
- Categorical encoding (one-hot, label)
- Feature scaling (standard, min-max)
- Outlier handling (removal, capping)
- Automated train-test splitting

### Model Training & Selection
- Cross-validation with appropriate strategies
- Performance metrics for classification/regression
- Model comparison visualizations
- Automatic problem type detection

### Hyperparameter Optimization
- Bayesian optimization with Optuna
- Efficient search strategies
- Convergence tracking
- Performance improvement analysis

### Model Interpretability
- Feature importance analysis
- SHAP (SHapley Additive exPlanations) values
- Model robustness testing
- Comprehensive evaluation metrics

## Development

### Local Testing

```bash
# Set up test environment
python -m venv automl_env
source automl_env/bin/activate  # Linux/Mac
# or
automl_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run directly for testing
python automl_server.py

# Test MCP protocol
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python automl_server.py
```

### Adding New Algorithms

1. Add the algorithm import and configuration in `train_baseline_models()`
2. Define hyperparameter space in `optimize_hyperparameters()`
3. Ensure compatibility with evaluation metrics
4. Update the algorithm lists in documentation

### Performance Optimization

- Large datasets are automatically sampled for SHAP analysis
- Cross-validation uses appropriate fold counts
- Memory cleanup utilities provided
- Efficient data structures used throughout

## Troubleshooting

### Tools Not Appearing
- Verify Docker image built successfully: `docker images`
- Check catalog and registry files for syntax errors
- Ensure Claude Desktop config includes custom catalog
- Restart Claude Desktop completely

### Memory Issues
- Use `cleanup_data()` to free memory between analyses
- Consider reducing `n_trials` for hyperparameter optimization
- For very large datasets, consider sampling before loading

### Model Training Failures
- Check data types and missing values
- Ensure target column exists and is properly formatted
- Verify sufficient data for train-test split
- Check logs for specific error messages

### Performance Issues
- Reduce cross-validation folds for faster training
- Limit number of baseline models
- Use smaller hyperparameter search spaces
- Consider feature selection for high-dimensional data

## Security Considerations

- All data processing done in isolated Docker container
- No external API calls or data transmission
- Models and data stored locally only
- Running as non-root user with minimal privileges
- Comprehensive input validation and error handling

## License

MIT License

## Contributing

Contributions welcome! Areas for improvement:
- Additional algorithms (CatBoost, Neural Networks)
- Advanced feature engineering
- Time series forecasting capabilities
- Automated feature selection
- Model ensembling techniques
- Integration with cloud ML services
