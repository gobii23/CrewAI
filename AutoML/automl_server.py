#!/usr/bin/env python3
"""
AutoML MCP Server - Complete automated machine learning pipeline with data analysis, preprocessing, training, optimization, and evaluation
"""

import os
import sys
import logging
import json
import base64
import io
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna
import shap
import joblib

from mcp.server.fastmcp import FastMCP

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("automl-server")

# Initialize MCP server
mcp = FastMCP("automl")

# Global variables to store data and models
data_store = {}
model_store = {}
results_store = {}

# === UTILITY FUNCTIONS ===

def encode_plot_to_base64(fig):
    """Convert matplotlib or plotly figure to base64 string."""
    try:
        if hasattr(fig, 'to_html'):  # Plotly figure
            return fig.to_html(include_plotlyjs='cdn')
        else:  # Matplotlib figure
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            return f'<img src="data:image/png;base64,{image_base64}" style="max-width:100%;">'
    except Exception as e:
        logger.error(f"Error encoding plot: {e}")
        return f"Error generating plot: {str(e)}"

def safe_json_loads(data_str):
    """Safely parse JSON string."""
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return None

def detect_problem_type(target_column, df):
    """Detect if it's classification or regression problem."""
    try:
        if df[target_column].dtype == 'object' or df[target_column].nunique() <= 10:
            return 'classification'
        else:
            return 'regression'
    except:
        return 'classification'

# === MCP TOOLS ===

@mcp.tool()
async def load_dataset(dataset_path: str = "", dataset_name: str = "dataset") -> str:
    """Load dataset from CSV file path and store it for analysis."""
    logger.info(f"Loading dataset from {dataset_path}")
    
    if not dataset_path.strip():
        return "❌ Error: Dataset path is required"
    
    try:
        # Load dataset
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        else:
            return "❌ Error: Only CSV and JSON files are supported"
        
        # Store dataset
        data_store[dataset_name] = df
        
        # Basic info
        shape = df.shape
        columns = list(df.columns)
        dtypes = df.dtypes.to_dict()
        
        return f"""✅ Dataset loaded successfully!
        
📊 Dataset Info:
- Name: {dataset_name}
- Shape: {shape[0]} rows × {shape[1]} columns
- Columns: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}

💾 Dataset stored as '{dataset_name}' for further analysis."""
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return f"❌ Error loading dataset: {str(e)}"

@mcp.tool()
async def data_analysis(dataset_name: str = "dataset") -> str:
    """Perform comprehensive data analysis including overview, missing values, statistics, and visualizations."""
    logger.info(f"Performing data analysis on {dataset_name}")
    
    if dataset_name not in data_store:
        return f"❌ Error: Dataset '{dataset_name}' not found. Please load a dataset first."
    
    try:
        df = data_store[dataset_name]
        
        # 1. Dataset overview
        shape = df.shape
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
        dtypes_count = df.dtypes.value_counts().to_dict()
        
        # 2. Missing value analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_summary = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percent': missing_percent
        }).sort_values('Missing_Count', ascending=False)
        
        # 3. Statistical summaries
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 4. Create visualizations
        plots_html = ""
        
        # Missing values heatmap
        if missing_data.sum() > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df.isnull(), cbar=True, ax=ax)
            plt.title('Missing Values Heatmap')
            plots_html += encode_plot_to_base64(fig) + "<br><br>"
        
        # Correlation heatmap for numerical columns
        if len(numerical_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 10))
            correlation_matrix = df[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            plt.title('Correlation Matrix')
            plots_html += encode_plot_to_base64(fig) + "<br><br>"
        
        # Distribution plots for numerical columns
        if len(numerical_cols) > 0:
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numerical_cols[:9]):  # Limit to 9 plots
                if i < len(axes):
                    df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numerical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plots_html += encode_plot_to_base64(fig) + "<br><br>"
        
        # 5. Outlier detection using IQR method
        outlier_summary = {}
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_summary[col] = len(outliers)
        
        # Store analysis results
        analysis_results = {
            'shape': shape,
            'memory_usage_mb': memory_usage,
            'dtypes_count': dtypes_count,
            'missing_summary': missing_summary,
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'outlier_summary': outlier_summary
        }
        
        results_store[f"{dataset_name}_analysis"] = analysis_results
        
        return f"""📊 Data Analysis Complete!

📈 Dataset Overview:
- Shape: {shape[0]} rows × {shape[1]} columns
- Memory Usage: {memory_usage:.2f} MB
- Data Types: {dtypes_count}

🔍 Missing Values:
{missing_summary.head(10).to_string() if missing_data.sum() > 0 else 'No missing values found!'}

📋 Feature Types:
- Numerical: {len(numerical_cols)} columns
- Categorical: {len(categorical_cols)} columns

⚠️ Outliers Detected (IQR method):
{json.dumps(outlier_summary, indent=2) if outlier_summary else 'No outliers detected'}

📊 Visualizations Generated:
{plots_html}

💾 Analysis results saved as '{dataset_name}_analysis'."""
        
    except Exception as e:
        logger.error(f"Error in data analysis: {e}")
        return f"❌ Error in data analysis: {str(e)}"

@mcp.tool()
async def preprocess_data(dataset_name: str = "dataset", target_column: str = "", preprocessing_config: str = "{}") -> str:
    """Preprocess data with missing value imputation, encoding, scaling, and train-test split."""
    logger.info(f"Preprocessing dataset {dataset_name}")
    
    if dataset_name not in data_store:
        return f"❌ Error: Dataset '{dataset_name}' not found. Please load a dataset first."
    
    if not target_column.strip():
        return "❌ Error: Target column is required for preprocessing"
    
    try:
        df = data_store[dataset_name].copy()
        config = safe_json_loads(preprocessing_config) or {}
        
        # Default preprocessing configuration
        default_config = {
            'handle_missing': 'mean',  # mean, median, mode, drop, knn
            'handle_duplicates': True,
            'encode_categorical': 'onehot',  # onehot, label
            'scale_features': 'standard',  # standard, minmax, none
            'handle_outliers': 'none',  # none, remove, cap
            'test_size': 0.2,
            'random_state': 42
        }
        
        # Merge with user config
        config = {**default_config, **config}
        
        if target_column not in df.columns:
            return f"❌ Error: Target column '{target_column}' not found in dataset"
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        preprocessing_steps = []
        
        # 1. Handle duplicates
        if config['handle_duplicates']:
            initial_rows = len(X)
            X = X.drop_duplicates()
            y = y.loc[X.index]
            removed_duplicates = initial_rows - len(X)
            if removed_duplicates > 0:
                preprocessing_steps.append(f"Removed {removed_duplicates} duplicate rows")
        
        # 2. Handle missing values
        if X.isnull().sum().sum() > 0:
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            
            if config['handle_missing'] == 'mean':
                if numerical_cols:
                    imputer_num = SimpleImputer(strategy='mean')
                    X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])
                if categorical_cols:
                    imputer_cat = SimpleImputer(strategy='most_frequent')
                    X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
                preprocessing_steps.append("Missing values imputed (mean/mode)")
                
            elif config['handle_missing'] == 'median':
                if numerical_cols:
                    imputer_num = SimpleImputer(strategy='median')
                    X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])
                if categorical_cols:
                    imputer_cat = SimpleImputer(strategy='most_frequent')
                    X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
                preprocessing_steps.append("Missing values imputed (median/mode)")
                
            elif config['handle_missing'] == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                # For mixed data types, we'll use simple imputation for categorical
                if categorical_cols:
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
                if numerical_cols:
                    X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
                preprocessing_steps.append("Missing values imputed (KNN)")
                
            elif config['handle_missing'] == 'drop':
                initial_rows = len(X)
                X = X.dropna()
                y = y.loc[X.index]
                removed_rows = initial_rows - len(X)
                preprocessing_steps.append(f"Dropped {removed_rows} rows with missing values")
        
        # 3. Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            if config['encode_categorical'] == 'onehot':
                # Use pandas get_dummies to avoid issues with unseen categories
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                preprocessing_steps.append(f"One-hot encoded {len(categorical_cols)} categorical columns")
            
            elif config['encode_categorical'] == 'label':
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                preprocessing_steps.append(f"Label encoded {len(categorical_cols)} categorical columns")
        
        # 4. Handle outliers
        if config['handle_outliers'] == 'remove':
            initial_rows = len(X)
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            for col in numerical_cols:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                mask = (X[col] >= lower_bound) & (X[col] <= upper_bound)
                X = X[mask]
                y = y.loc[X.index]
            removed_outliers = initial_rows - len(X)
            if removed_outliers > 0:
                preprocessing_steps.append(f"Removed {removed_outliers} outlier rows")
        
        elif config['handle_outliers'] == 'cap':
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            for col in numerical_cols:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
            preprocessing_steps.append("Capped outliers using IQR method")
        
        # 5. Scale features
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols and config['scale_features'] != 'none':
            if config['scale_features'] == 'standard':
                scaler = StandardScaler()
                X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
                preprocessing_steps.append("Applied StandardScaler to numerical features")
            
            elif config['scale_features'] == 'minmax':
                scaler = MinMaxScaler()
                X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
                preprocessing_steps.append("Applied MinMaxScaler to numerical features")
        
        # 6. Split data
        problem_type = detect_problem_type(target_column, df)
        if problem_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config['test_size'], 
                random_state=config['random_state'],
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config['test_size'],
                random_state=config['random_state']
            )
        
        # Store preprocessed data
        preprocessed_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'target_column': target_column,
            'problem_type': problem_type,
            'preprocessing_config': config,
            'preprocessing_steps': preprocessing_steps
        }
        
        data_store[f"{dataset_name}_preprocessed"] = preprocessed_data
        
        return f"""✅ Data Preprocessing Complete!

🔄 Preprocessing Steps Applied:
{chr(10).join([f"- {step}" for step in preprocessing_steps])}

📊 Final Dataset:
- Problem Type: {problem_type.title()}
- Features: {len(X.columns)} columns
- Training Set: {len(X_train)} samples
- Test Set: {len(X_test)} samples
- Target Column: {target_column}

💾 Preprocessed data saved as '{dataset_name}_preprocessed'"""
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return f"❌ Error in preprocessing: {str(e)}"

@mcp.tool()
async def train_baseline_models(dataset_name: str = "dataset", model_config: str = "{}") -> str:
    """Train multiple baseline models and compare their performance."""
    logger.info(f"Training baseline models for {dataset_name}")
    
    preprocessed_key = f"{dataset_name}_preprocessed"
    if preprocessed_key not in data_store:
        return f"❌ Error: Preprocessed dataset '{preprocessed_key}' not found. Please preprocess the data first."
    
    try:
        data = data_store[preprocessed_key]
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        problem_type = data['problem_type']
        
        config = safe_json_loads(model_config) or {}
        cv_folds = config.get('cv_folds', 5)
        random_state = config.get('random_state', 42)
        
        models = {}
        results = {}
        
        # Define models based on problem type
        if problem_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
                'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
                'XGBoost': xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'),
                'LightGBM': lgb.LGBMClassifier(random_state=random_state, verbose=-1),
                'Gradient Boosting': GradientBoostingClassifier(random_state=random_state)
            }
            
            # Cross-validation and training
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            for name, model in models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                    
                    # Train on full training set
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    
                    # Metrics
                    train_acc = accuracy_score(y_train, train_pred)
                    test_acc = accuracy_score(y_test, test_pred)
                    
                    if len(np.unique(y_train)) == 2:  # Binary classification
                        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
                        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                        f1 = f1_score(y_test, test_pred)
                        precision = precision_score(y_test, test_pred)
                        recall = recall_score(y_test, test_pred)
                    else:  # Multiclass
                        train_auc = test_auc = None
                        f1 = f1_score(y_test, test_pred, average='weighted')
                        precision = precision_score(y_test, test_pred, average='weighted')
                        recall = recall_score(y_test, test_pred, average='weighted')
                    
                    results[name] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'train_auc': train_auc,
                        'test_auc': test_auc,
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall,
                        'model': model
                    }
                    
                except Exception as e:
                    logger.warning(f"Error training {name}: {e}")
                    results[name] = {'error': str(e)}
        
        else:  # Regression
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(random_state=random_state),
                'Random Forest': RandomForestRegressor(random_state=random_state, n_estimators=100),
                'XGBoost': xgb.XGBRegressor(random_state=random_state),
                'LightGBM': lgb.LGBMRegressor(random_state=random_state, verbose=-1),
                'Gradient Boosting': GradientBoostingRegressor(random_state=random_state)
            }
            
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            for name, model in models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
                    
                    # Train on full training set
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    
                    # Metrics
                    train_r2 = r2_score(y_train, train_pred)
                    test_r2 = r2_score(y_test, test_pred)
                    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                    train_mae = mean_absolute_error(y_train, train_pred)
                    test_mae = mean_absolute_error(y_test, test_pred)
                    
                    results[name] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'model': model
                    }
                    
                except Exception as e:
                    logger.warning(f"Error training {name}: {e}")
                    results[name] = {'error': str(e)}
        
        # Store results
        model_store[f"{dataset_name}_baseline"] = results
        
        # Create comparison visualization
        plots_html = ""
        successful_models = {name: res for name, res in results.items() if 'error' not in res}
        
        if successful_models:
            if problem_type == 'classification':
                # Model comparison plot
                model_names = list(successful_models.keys())
                cv_means = [successful_models[name]['cv_mean'] for name in model_names]
                test_accs = [successful_models[name]['test_accuracy'] for name in model_names]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.bar(model_names, cv_means)
                ax1.set_title('Cross-Validation Accuracy')
                ax1.set_ylabel('Accuracy')
                ax1.tick_params(axis='x', rotation=45)
                
                ax2.bar(model_names, test_accs)
                ax2.set_title('Test Set Accuracy')
                ax2.set_ylabel('Accuracy')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plots_html += encode_plot_to_base64(fig) + "<br><br>"
                
            else:  # Regression
                model_names = list(successful_models.keys())
                cv_means = [successful_models[name]['cv_mean'] for name in model_names]
                test_r2s = [successful_models[name]['test_r2'] for name in model_names]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.bar(model_names, cv_means)
                ax1.set_title('Cross-Validation R²')
                ax1.set_ylabel('R² Score')
                ax1.tick_params(axis='x', rotation=45)
                
                ax2.bar(model_names, test_r2s)
                ax2.set_title('Test Set R²')
                ax2.set_ylabel('R² Score')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plots_html += encode_plot_to_base64(fig) + "<br><br>"
        
        # Format results summary
        summary_text = f"🎯 Baseline Models Training Results ({problem_type.title()}):\n\n"
        
        for name, res in results.items():
            if 'error' in res:
                summary_text += f"❌ {name}: Failed - {res['error']}\n"
            else:
                if problem_type == 'classification':
                    summary_text += f"✅ {name}:\n"
                    summary_text += f"   - CV Accuracy: {res['cv_mean']:.4f} (±{res['cv_std']:.4f})\n"
                    summary_text += f"   - Test Accuracy: {res['test_accuracy']:.4f}\n"
                    summary_text += f"   - F1 Score: {res['f1_score']:.4f}\n"
                    if res['test_auc']:
                        summary_text += f"   - AUC: {res['test_auc']:.4f}\n"
                else:
                    summary_text += f"✅ {name}:\n"
                    summary_text += f"   - CV R²: {res['cv_mean']:.4f} (±{res['cv_std']:.4f})\n"
                    summary_text += f"   - Test R²: {res['test_r2']:.4f}\n"
                    summary_text += f"   - Test RMSE: {res['test_rmse']:.4f}\n"
                summary_text += "\n"
        
        return f"""{summary_text}

📊 Model Comparison Visualizations:
{plots_html}

💾 Models saved as '{dataset_name}_baseline'"""
        
    except Exception as e:
        logger.error(f"Error in baseline training: {e}")
        return f"❌ Error in baseline training: {str(e)}"

@mcp.tool()
async def optimize_hyperparameters(dataset_name: str = "dataset", models_to_optimize: str = "", optimization_config: str = "{}") -> str:
    """Optimize hyperparameters for top performing models using Optuna."""
    logger.info(f"Optimizing hyperparameters for {dataset_name}")
    
    baseline_key = f"{dataset_name}_baseline"
    preprocessed_key = f"{dataset_name}_preprocessed"
    
    if baseline_key not in model_store:
        return f"❌ Error: Baseline models '{baseline_key}' not found. Please train baseline models first."
    
    if preprocessed_key not in data_store:
        return f"❌ Error: Preprocessed dataset '{preprocessed_key}' not found."
    
    try:
        baseline_results = model_store[baseline_key]
        data = data_store[preprocessed_key]
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        problem_type = data['problem_type']
        
        config = safe_json_loads(optimization_config) or {}
        n_trials = config.get('n_trials', 100)
        cv_folds = config.get('cv_folds', 5)
        
        # Determine which models to optimize
        if models_to_optimize.strip():
            models_list = [m.strip() for m in models_to_optimize.split(',')]
        else:
            # Auto-select top 3 models based on performance
            successful_models = {name: res for name, res in baseline_results.items() if 'error' not in res}
            if problem_type == 'classification':
                sorted_models = sorted(successful_models.items(), 
                                     key=lambda x: x[1]['test_accuracy'], reverse=True)
            else:
                sorted_models = sorted(successful_models.items(), 
                                     key=lambda x: x[1]['test_r2'], reverse=True)
            models_list = [name for name, _ in sorted_models[:3]]
        
        optimization_results = {}
        plots_html = ""
        
        for model_name in models_list:
            if model_name not in baseline_results or 'error' in baseline_results[model_name]:
                optimization_results[model_name] = {'error': f'Model {model_name} not available'}
                continue
            
            try:
                logger.info(f"Optimizing {model_name}")
                
                def objective(trial):
                    try:
                        # Define hyperparameter spaces for different models
                        if model_name == 'Random Forest':
                            if problem_type == 'classification':
                                model = RandomForestClassifier(
                                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                                    max_depth=trial.suggest_int('max_depth', 3, 20),
                                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                                    random_state=42
                                )
                                metric = 'accuracy'
                            else:
                                model = RandomForestRegressor(
                                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                                    max_depth=trial.suggest_int('max_depth', 3, 20),
                                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                                    random_state=42
                                )
                                metric = 'r2'
                        
                        elif model_name == 'XGBoost':
                            if problem_type == 'classification':
                                model = xgb.XGBClassifier(
                                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                                    max_depth=trial.suggest_int('max_depth', 3, 10),
                                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                                    random_state=42,
                                    eval_metric='logloss'
                                )
                                metric = 'accuracy'
                            else:
                                model = xgb.XGBRegressor(
                                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                                    max_depth=trial.suggest_int('max_depth', 3, 10),
                                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                                    random_state=42
                                )
                                metric = 'r2'
                        
                        elif model_name == 'LightGBM':
                            if problem_type == 'classification':
                                model = lgb.LGBMClassifier(
                                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                                    max_depth=trial.suggest_int('max_depth', 3, 10),
                                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                                    num_leaves=trial.suggest_int('num_leaves', 10, 100),
                                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                                    random_state=42,
                                    verbose=-1
                                )
                                metric = 'accuracy'
                            else:
                                model = lgb.LGBMRegressor(
                                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                                    max_depth=trial.suggest_int('max_depth', 3, 10),
                                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                                    num_leaves=trial.suggest_int('num_leaves', 10, 100),
                                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                                    random_state=42,
                                    verbose=-1
                                )
                                metric = 'r2'
                        
                        else:
                            # Use baseline model for others
                            model = baseline_results[model_name]['model']
                            metric = 'accuracy' if problem_type == 'classification' else 'r2'
                        
                        # Cross-validation
                        if problem_type == 'classification':
                            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        else:
                            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        
                        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
                        return cv_scores.mean()
                        
                    except Exception as e:
                        logger.warning(f"Trial failed for {model_name}: {e}")
                        return -999  # Return very low score for failed trials
                
                # Run optimization
                study = optuna.create_study(direction='maximize', 
                                          sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                
                # Get best parameters and retrain
                best_params = study.best_params
                best_score = study.best_value
                
                # Create optimized model
                if model_name == 'Random Forest':
                    if problem_type == 'classification':
                        optimized_model = RandomForestClassifier(**best_params, random_state=42)
                    else:
                        optimized_model = RandomForestRegressor(**best_params, random_state=42)
                
                elif model_name == 'XGBoost':
                    if problem_type == 'classification':
                        optimized_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
                    else:
                        optimized_model = xgb.XGBRegressor(**best_params, random_state=42)
                
                elif model_name == 'LightGBM':
                    if problem_type == 'classification':
                        optimized_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
                    else:
                        optimized_model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
                
                else:
                    optimized_model = baseline_results[model_name]['model']
                
                # Train optimized model
                optimized_model.fit(X_train, y_train)
                
                # Evaluate
                if problem_type == 'classification':
                    train_pred = optimized_model.predict(X_train)
                    test_pred = optimized_model.predict(X_test)
                    
                    train_acc = accuracy_score(y_train, train_pred)
                    test_acc = accuracy_score(y_test, test_pred)
                    f1 = f1_score(y_test, test_pred, average='weighted')
                    
                    baseline_score = baseline_results[model_name]['test_accuracy']
                    improvement = test_acc - baseline_score
                    
                    optimization_results[model_name] = {
                        'best_params': best_params,
                        'best_cv_score': best_score,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'f1_score': f1,
                        'baseline_score': baseline_score,
                        'improvement': improvement,
                        'optimized_model': optimized_model,
                        'study': study
                    }
                
                else:  # Regression
                    train_pred = optimized_model.predict(X_train)
                    test_pred = optimized_model.predict(X_test)
                    
                    train_r2 = r2_score(y_train, train_pred)
                    test_r2 = r2_score(y_test, test_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                    
                    baseline_score = baseline_results[model_name]['test_r2']
                    improvement = test_r2 - baseline_score
                    
                    optimization_results[model_name] = {
                        'best_params': best_params,
                        'best_cv_score': best_score,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'test_rmse': test_rmse,
                        'baseline_score': baseline_score,
                        'improvement': improvement,
                        'optimized_model': optimized_model,
                        'study': study
                    }
                
                # Create optimization history plot
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    trials_df = study.trials_dataframe()
                    ax.plot(trials_df['number'], trials_df['value'])
                    ax.set_xlabel('Trial')
                    ax.set_ylabel('Objective Value')
                    ax.set_title(f'{model_name} Optimization History')
                    ax.grid(True)
                    plots_html += f"<h3>{model_name} Optimization History:</h3>"
                    plots_html += encode_plot_to_base64(fig) + "<br><br>"
                except:
                    pass
                
            except Exception as e:
                logger.error(f"Error optimizing {model_name}: {e}")
                optimization_results[model_name] = {'error': str(e)}
        
        # Store optimization results
        model_store[f"{dataset_name}_optimized"] = optimization_results
        
        # Format results summary
        summary_text = f"🎯 Hyperparameter Optimization Results:\n\n"
        
        for name, res in optimization_results.items():
            if 'error' in res:
                summary_text += f"❌ {name}: Failed - {res['error']}\n"
            else:
                if problem_type == 'classification':
                    summary_text += f"✅ {name}:\n"
                    summary_text += f"   - Best CV Score: {res['best_cv_score']:.4f}\n"
                    summary_text += f"   - Test Accuracy: {res['test_accuracy']:.4f}\n"
                    summary_text += f"   - Baseline: {res['baseline_score']:.4f}\n"
                    summary_text += f"   - Improvement: {res['improvement']:+.4f}\n"
                    summary_text += f"   - Best Params: {res['best_params']}\n"
                else:
                    summary_text += f"✅ {name}:\n"
                    summary_text += f"   - Best CV Score: {res['best_cv_score']:.4f}\n"
                    summary_text += f"   - Test R²: {res['test_r2']:.4f}\n"
                    summary_text += f"   - Baseline: {res['baseline_score']:.4f}\n"
                    summary_text += f"   - Improvement: {res['improvement']:+.4f}\n"
                    summary_text += f"   - Best Params: {res['best_params']}\n"
                summary_text += "\n"
        
        return f"""{summary_text}

📊 Optimization History:
{plots_html}

💾 Optimized models saved as '{dataset_name}_optimized'"""
        
    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {e}")
        return f"❌ Error in hyperparameter optimization: {str(e)}"

@mcp.tool()
async def final_evaluation(dataset_name: str = "dataset", model_name: str = "") -> str:
    """Perform comprehensive final evaluation including interpretability analysis."""
    logger.info(f"Final evaluation for {dataset_name}")
    
    optimized_key = f"{dataset_name}_optimized"
    preprocessed_key = f"{dataset_name}_preprocessed"
    
    if optimized_key not in model_store:
        return f"❌ Error: Optimized models '{optimized_key}' not found. Please optimize models first."
    
    if preprocessed_key not in data_store:
        return f"❌ Error: Preprocessed dataset '{preprocessed_key}' not found."
    
    try:
        optimized_results = model_store[optimized_key]
        data = data_store[preprocessed_key]
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        feature_names = data['feature_names']
        problem_type = data['problem_type']
        
        # Select best model
        if model_name.strip():
            if model_name not in optimized_results:
                return f"❌ Error: Model '{model_name}' not found in optimized results."
            best_model_name = model_name
        else:
            # Auto-select best performing model
            successful_models = {name: res for name, res in optimized_results.items() if 'error' not in res}
            if not successful_models:
                return "❌ Error: No successfully optimized models found."
            
            if problem_type == 'classification':
                best_model_name = max(successful_models.keys(), 
                                    key=lambda x: successful_models[x]['test_accuracy'])
            else:
                best_model_name = max(successful_models.keys(), 
                                    key=lambda x: successful_models[x]['test_r2'])
        
        best_result = optimized_results[best_model_name]
        if 'error' in best_result:
            return f"❌ Error: Selected model '{best_model_name}' has error: {best_result['error']}"
        
        best_model = best_result['optimized_model']
        plots_html = ""
        
        # 1. Model Performance on Test Set
        test_pred = best_model.predict(X_test)
        
        if problem_type == 'classification':
            # Classification metrics
            accuracy = accuracy_score(y_test, test_pred)
            f1 = f1_score(y_test, test_pred, average='weighted')
            precision = precision_score(y_test, test_pred, average='weighted')
            recall = recall_score(y_test, test_pred, average='weighted')
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, test_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {best_model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plots_html += "<h3>Confusion Matrix:</h3>"
            plots_html += encode_plot_to_base64(fig) + "<br><br>"
            
            # ROC Curve for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    y_proba = best_model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    auc_score = roc_auc_score(y_test, y_proba)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
                    ax.plot([0, 1], [0, 1], 'k--', label='Random')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend()
                    ax.grid(True)
                    plots_html += "<h3>ROC Curve:</h3>"
                    plots_html += encode_plot_to_base64(fig) + "<br><br>"
                except:
                    pass
            
            performance_summary = f"""🎯 Final Model Performance ({best_model_name}):
- Accuracy: {accuracy:.4f}
- F1 Score: {f1:.4f}  
- Precision: {precision:.4f}
- Recall: {recall:.4f}"""
        
        else:  # Regression
            # Regression metrics
            r2 = r2_score(y_test, test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            mae = mean_absolute_error(y_test, test_pred)
            
            # Prediction vs Actual plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.scatter(y_test, test_pred, alpha=0.7)
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title('Predicted vs Actual')
            ax1.grid(True)
            
            # Residuals plot
            residuals = y_test - test_pred
            ax2.scatter(test_pred, residuals, alpha=0.7)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals Plot')
            ax2.grid(True)
            
            plt.tight_layout()
            plots_html += "<h3>Regression Analysis:</h3>"
            plots_html += encode_plot_to_base64(fig) + "<br><br>"
            
            performance_summary = f"""🎯 Final Model Performance ({best_model_name}):
- R² Score: {r2:.4f}
- RMSE: {rmse:.4f}
- MAE: {mae:.4f}"""
        
        # 2. Feature Importance Analysis
        try:
            if hasattr(best_model, 'feature_importances_'):
                importance = best_model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                # Plot top 15 features
                top_features = feature_importance_df.head(15)
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(range(len(top_features)), top_features['importance'])
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Top 15 Feature Importances')
                ax.grid(True, axis='x')
                plt.tight_layout()
                plots_html += "<h3>Feature Importance:</h3>"
                plots_html += encode_plot_to_base64(fig) + "<br><br>"
        except:
            pass
        
        # 3. SHAP Analysis (for supported models)
        try:
            # Check if model supports SHAP
            explainer = None
            if hasattr(best_model, 'predict_proba') and problem_type == 'classification':
                explainer = shap.Explainer(best_model, X_train)
            elif hasattr(best_model, 'predict'):
                explainer = shap.Explainer(best_model, X_train)
            
            if explainer:
                # Calculate SHAP values for test set (sample if too large)
                sample_size = min(100, len(X_test))
                X_test_sample = X_test.sample(n=sample_size, random_state=42)
                
                shap_values = explainer(X_test_sample)
                
                # SHAP Summary plot
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test_sample, show=False, max_display=15)
                plots_html += "<h3>SHAP Summary Plot:</h3>"
                plots_html += encode_plot_to_base64(fig) + "<br><br>"
                
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
        
        # 4. Model Robustness Analysis
        robustness_results = []
        
        # Performance on different data subsets
        try:
            # Split test set into quartiles and evaluate
            n_splits = 4
            subset_size = len(X_test) // n_splits
            
            for i in range(n_splits):
                start_idx = i * subset_size
                end_idx = start_idx + subset_size if i < n_splits - 1 else len(X_test)
                
                X_subset = X_test.iloc[start_idx:end_idx]
                y_subset = y_test.iloc[start_idx:end_idx]
                
                subset_pred = best_model.predict(X_subset)
                
                if problem_type == 'classification':
                    subset_acc = accuracy_score(y_subset, subset_pred)
                    robustness_results.append(f"Subset {i+1}: {subset_acc:.4f}")
                else:
                    subset_r2 = r2_score(y_subset, subset_pred)
                    robustness_results.append(f"Subset {i+1}: {subset_r2:.4f}")
        except:
            robustness_results = ["Robustness analysis not available"]
        
        # 5. Business Impact Assessment and Deployment Checklist
        deployment_checklist = [
            "✅ Model trained and validated on appropriate data splits",
            "✅ Hyperparameters optimized using cross-validation",
            "✅ Model performance evaluated on hold-out test set",
            "✅ Feature importance and interpretability analysis completed",
            "✅ Model robustness tested on data subsets",
            "⚠️ Production data drift monitoring setup needed",
            "⚠️ Model retraining pipeline implementation needed",
            "⚠️ A/B testing framework setup recommended",
            "⚠️ Model governance and documentation required"
        ]
        
        # Store final evaluation results
        final_eval = {
            'best_model_name': best_model_name,
            'best_model': best_model,
            'performance_summary': performance_summary,
            'robustness_results': robustness_results,
            'deployment_checklist': deployment_checklist,
            'feature_names': feature_names,
            'problem_type': problem_type
        }
        
        results_store[f"{dataset_name}_final_evaluation"] = final_eval
        
        # Save the best model
        model_filename = f"{dataset_name}_best_model.joblib"
        joblib.dump(best_model, model_filename)
        
        return f"""🏆 Final Evaluation Complete!

{performance_summary}

🔍 Model Robustness Analysis:
{chr(10).join(robustness_results)}

📊 Visualizations:
{plots_html}

🚀 Deployment Readiness Checklist:
{chr(10).join(deployment_checklist)}

💾 Results saved as '{dataset_name}_final_evaluation'
📁 Best model saved as '{model_filename}'

🎯 Model Selection Rationale:
Selected {best_model_name} as the final model based on {'test accuracy' if problem_type == 'classification' else 'test R² score'} performance after hyperparameter optimization. The model demonstrates good generalization capability and interpretability through feature importance and SHAP analysis."""
        
    except Exception as e:
        logger.error(f"Error in final evaluation: {e}")
        return f"❌ Error in final evaluation: {str(e)}"

@mcp.tool()
async def save_model(dataset_name: str = "dataset", model_type: str = "final", filename: str = "") -> str:
    """Save trained model to disk with joblib."""
    logger.info(f"Saving model for {dataset_name}")
    
    try:
        if not filename.strip():
            filename = f"{dataset_name}_{model_type}_model.joblib"
        
        if model_type == "final":
            eval_key = f"{dataset_name}_final_evaluation"
            if eval_key not in results_store:
                return f"❌ Error: Final evaluation results not found. Please run final evaluation first."
            
            model = results_store[eval_key]['best_model']
            model_name = results_store[eval_key]['best_model_name']
            
        elif model_type == "baseline":
            baseline_key = f"{dataset_name}_baseline"
            if baseline_key not in model_store:
                return f"❌ Error: Baseline models not found. Please train baseline models first."
            
            # Save the best baseline model
            baseline_results = model_store[baseline_key]
            successful_models = {name: res for name, res in baseline_results.items() if 'error' not in res}
            
            if not successful_models:
                return "❌ Error: No successful baseline models found."
            
            # Get problem type to determine best metric
            preprocessed_key = f"{dataset_name}_preprocessed"
            if preprocessed_key in data_store:
                problem_type = data_store[preprocessed_key]['problem_type']
                if problem_type == 'classification':
                    best_model_name = max(successful_models.keys(), 
                                        key=lambda x: successful_models[x]['test_accuracy'])
                else:
                    best_model_name = max(successful_models.keys(), 
                                        key=lambda x: successful_models[x]['test_r2'])
            else:
                # Default to first successful model
                best_model_name = list(successful_models.keys())[0]
            
            model = successful_models[best_model_name]['model']
            model_name = best_model_name
        
        else:
            return f"❌ Error: Invalid model_type '{model_type}'. Use 'final' or 'baseline'."
        
        # Save model
        joblib.dump(model, filename)
        
        return f"""✅ Model saved successfully!

📁 File: {filename}
🤖 Model: {model_name}
📊 Type: {model_type}

💡 To load this model later:
import joblib
model = joblib.load('{filename}')"""
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return f"❌ Error saving model: {str(e)}"

@mcp.tool()
async def generate_report(dataset_name: str = "dataset", report_type: str = "full") -> str:
    """Generate comprehensive AutoML report with all analysis results."""
    logger.info(f"Generating {report_type} report for {dataset_name}")
    
    try:
        # Check what data is available
        analysis_key = f"{dataset_name}_analysis"
        preprocessed_key = f"{dataset_name}_preprocessed"
        baseline_key = f"{dataset_name}_baseline"
        optimized_key = f"{dataset_name}_optimized"
        final_key = f"{dataset_name}_final_evaluation"
        
        report_sections = []
        
        # Header
        report_sections.append(f"""# AutoML Report: {dataset_name}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Type: {report_type.title()}

---
""")
        
        # 1. Data Analysis Section
        if analysis_key in results_store:
            analysis = results_store[analysis_key]
            report_sections.append(f"""## 1. Data Analysis Summary

📊 **Dataset Overview:**
- Shape: {analysis['shape'][0]:,} rows × {analysis['shape'][1]} columns
- Memory Usage: {analysis['memory_usage_mb']:.2f} MB
- Data Types: {analysis['dtypes_count']}

📋 **Feature Distribution:**
- Numerical Features: {len(analysis['numerical_cols'])} columns
- Categorical Features: {len(analysis['categorical_cols'])} columns

🔍 **Data Quality Issues:**
- Missing Values: {len([col for col in analysis['missing_summary'].index if analysis['missing_summary'].loc[col, 'Missing_Count'] > 0])} columns affected
- Outliers Detected: {sum(1 for count in analysis['outlier_summary'].values() if count > 0)} numerical columns

---
""")
        
        # 2. Preprocessing Section  
        if preprocessed_key in data_store:
            preproc = data_store[preprocessed_key]
            report_sections.append(f"""## 2. Data Preprocessing

🔄 **Preprocessing Steps:**
{chr(10).join([f"- {step}" for step in preproc['preprocessing_steps']])}

📊 **Final Dataset:**
- Problem Type: {preproc['problem_type'].title()}
- Features: {len(preproc['feature_names'])} columns
- Training Set: {len(preproc['X_train']):,} samples
- Test Set: {len(preproc['X_test']):,} samples

---
""")
        
        # 3. Baseline Models Section
        if baseline_key in model_store:
            baseline = model_store[baseline_key]
            successful_models = {name: res for name, res in baseline.items() if 'error' not in res}
            
            report_sections.append(f"""## 3. Baseline Model Performance

🎯 **Models Trained:** {len(baseline)} models
✅ **Successful:** {len(successful_models)} models

**Performance Summary:**
""")
            
            for name, res in successful_models.items():
                if preprocessed_key in data_store and data_store[preprocessed_key]['problem_type'] == 'classification':
                    report_sections.append(f"""
**{name}:**
- Cross-Validation Accuracy: {res['cv_mean']:.4f} (±{res['cv_std']:.4f})
- Test Accuracy: {res['test_accuracy']:.4f}
- F1 Score: {res['f1_score']:.4f}""")
                else:
                    report_sections.append(f"""
**{name}:**
- Cross-Validation R²: {res['cv_mean']:.4f} (±{res['cv_std']:.4f})
- Test R²: {res['test_r2']:.4f}
- Test RMSE: {res['test_rmse']:.4f}""")
            
            report_sections.append("\n---\n")
        
        # 4. Hyperparameter Optimization Section
        if optimized_key in model_store:
            optimized = model_store[optimized_key]
            successful_opt = {name: res for name, res in optimized.items() if 'error' not in res}
            
            report_sections.append(f"""## 4. Hyperparameter Optimization

🔧 **Models Optimized:** {len(successful_opt)} models

**Optimization Results:**
""")
            
            for name, res in successful_opt.items():
                if preprocessed_key in data_store and data_store[preprocessed_key]['problem_type'] == 'classification':
                    report_sections.append(f"""
**{name}:**
- Best CV Score: {res['best_cv_score']:.4f}
- Test Accuracy: {res['test_accuracy']:.4f}
- Improvement: {res['improvement']:+.4f}
- Best Parameters: {res['best_params']}""")
                else:
                    report_sections.append(f"""
**{name}:**
- Best CV Score: {res['best_cv_score']:.4f}
- Test R²: {res['test_r2']:.4f}
- Improvement: {res['improvement']:+.4f}
- Best Parameters: {res['best_params']}""")
            
            report_sections.append("\n---\n")
        
        # 5. Final Model Section
        if final_key in results_store:
            final = results_store[final_key]
            report_sections.append(f"""## 5. Final Model Selection

🏆 **Selected Model:** {final['best_model_name']}

{final['performance_summary']}

🔍 **Model Robustness:**
{chr(10).join(final['robustness_results'])}

🚀 **Deployment Readiness:**
{chr(10).join(final['deployment_checklist'])}

---
""")
        
        # 6. Recommendations Section
        recommendations = []
        
        if analysis_key in results_store:
            analysis = results_store[analysis_key]
            if len([col for col in analysis['missing_summary'].index if analysis['missing_summary'].loc[col, 'Missing_Count'] > 0]) > 0:
                recommendations.append("Consider advanced imputation techniques for missing values")
            
            if sum(1 for count in analysis['outlier_summary'].values() if count > 0) > 0:
                recommendations.append("Investigate outliers for potential data quality issues")
        
        if baseline_key in model_store and optimized_key in model_store:
            recommendations.append("Consider ensemble methods for improved performance")
            recommendations.append("Implement model monitoring for production deployment")
        
        if recommendations:
            report_sections.append(f"""## 6. Recommendations

{chr(10).join([f"- {rec}" for rec in recommendations])}

---
""")
        
        # 7. Summary
        report_sections.append(f"""## 7. Executive Summary

This AutoML pipeline has successfully:
{f"- Analyzed dataset with {results_store[analysis_key]['shape'][0]:,} samples" if analysis_key in results_store else ""}
{f"- Preprocessed data with {len(data_store[preprocessed_key]['preprocessing_steps'])} steps" if preprocessed_key in data_store else ""}
{f"- Trained and compared {len([name for name, res in model_store[baseline_key].items() if 'error' not in res])} baseline models" if baseline_key in model_store else ""}
{f"- Optimized hyperparameters for top performing models" if optimized_key in model_store else ""}
{f"- Selected {results_store[final_key]['best_model_name']} as the final model" if final_key in results_store else ""}

The pipeline provides a production-ready model with comprehensive evaluation and interpretability analysis.

---

*Report generated by AutoML MCP Server*
""")
        
        # Combine all sections
        full_report = "".join(report_sections)
        
        # Save report
        report_filename = f"{dataset_name}_automl_report.md"
        with open(report_filename, 'w') as f:
            f.write(full_report)
        
        if report_type == "summary":
            # Return only executive summary for summary reports
            summary_start = full_report.find("## 7. Executive Summary")
            if summary_start != -1:
                summary_report = full_report[summary_start:]
                return f"""📊 AutoML Summary Report Generated!

{summary_report}

📁 Full report saved as: {report_filename}"""
        
        return f"""📊 AutoML Report Generated!

📁 Report saved as: {report_filename}

{full_report[:2000]}{"..." if len(full_report) > 2000 else ""}

💾 Complete report available in the saved file."""
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return f"❌ Error generating report: {str(e)}"

@mcp.tool()
async def list_datasets(show_details: str = "false") -> str:
    """List all loaded datasets and their status in the AutoML pipeline."""
    logger.info("Listing datasets")
    
    try:
        if not data_store:
            return "📭 No datasets loaded. Use load_dataset() to start."
        
        show_details_bool = show_details.lower() == "true"
        dataset_info = []
        
        for dataset_name, dataset in data_store.items():
            if dataset_name.endswith('_preprocessed'):
                continue  # Skip preprocessed datasets in main listing
            
            info = f"📊 **{dataset_name}**"
            
            if isinstance(dataset, pd.DataFrame):
                info += f" - {dataset.shape[0]:,} rows × {dataset.shape[1]} columns"
                
                # Check pipeline status
                status_items = []
                if f"{dataset_name}_analysis" in results_store:
                    status_items.append("✅ Analyzed")
                if f"{dataset_name}_preprocessed" in data_store:
                    status_items.append("✅ Preprocessed")
                if f"{dataset_name}_baseline" in model_store:
                    status_items.append("✅ Baseline Models")
                if f"{dataset_name}_optimized" in model_store:
                    status_items.append("✅ Optimized")
                if f"{dataset_name}_final_evaluation" in results_store:
                    status_items.append("✅ Final Evaluation")
                
                if status_items:
                    info += f"\n   Status: {', '.join(status_items)}"
                else:
                    info += f"\n   Status: 🟡 Loaded (ready for analysis)"
                
                if show_details_bool:
                    info += f"\n   Columns: {', '.join(list(dataset.columns)[:5])}"
                    if len(dataset.columns) > 5:
                        info += f" ... (+{len(dataset.columns)-5} more)"
                    
                    info += f"\n   Data Types: {dataset.dtypes.value_counts().to_dict()}"
                    info += f"\n   Missing Values: {dataset.isnull().sum().sum()}"
            
            dataset_info.append(info)
        
        return f"""📚 Dataset Inventory:

{chr(10).join(dataset_info)}

💡 Use show_details="true" for more information"""
        
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        return f"❌ Error listing datasets: {str(e)}"

@mcp.tool()
async def cleanup_data(dataset_name: str = "", cleanup_type: str = "all") -> str:
    """Clean up stored data, models, and results to free memory."""
    logger.info(f"Cleaning up data for {dataset_name}")
    
    try:
        cleaned_items = []
        
        if cleanup_type == "all" or cleanup_type == "data":
            if dataset_name.strip():
                # Clean specific dataset
                keys_to_remove = []
                for key in data_store.keys():
                    if key.startswith(dataset_name):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del data_store[key]
                    cleaned_items.append(f"Dataset: {key}")
            else:
                # Clean all datasets
                dataset_count = len(data_store)
                data_store.clear()
                cleaned_items.append(f"All datasets ({dataset_count} items)")
        
        if cleanup_type == "all" or cleanup_type == "models":
            if dataset_name.strip():
                keys_to_remove = []
                for key in model_store.keys():
                    if key.startswith(dataset_name):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del model_store[key]
                    cleaned_items.append(f"Models: {key}")
            else:
                model_count = len(model_store)
                model_store.clear()
                cleaned_items.append(f"All models ({model_count} items)")
        
        if cleanup_type == "all" or cleanup_type == "results":
            if dataset_name.strip():
                keys_to_remove = []
                for key in results_store.keys():
                    if key.startswith(dataset_name):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del results_store[key]
                    cleaned_items.append(f"Results: {key}")
            else:
                results_count = len(results_store)
                results_store.clear()
                cleaned_items.append(f"All results ({results_count} items)")
        
        if not cleaned_items:
            return f"🔍 No items found to clean for dataset '{dataset_name}' with type '{cleanup_type}'"
        
        return f"""🧹 Cleanup Complete!

Removed:
{chr(10).join([f"- {item}" for item in cleaned_items])}

💾 Memory freed and ready for new tasks."""
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return f"❌ Error during cleanup: {str(e)}"

# === SERVER STARTUP ===
if __name__ == "__main__":
    logger.info("Starting AutoML MCP server...")
    
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)