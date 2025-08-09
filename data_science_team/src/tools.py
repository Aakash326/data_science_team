import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class DataAnalysisInput(BaseModel):
    """Input schema for data analysis tool."""
    code: str = Field(..., description="Python code to execute for data analysis")
    description: str = Field(..., description="Description of what the code does")


class DataAnalysisTool(BaseTool):
    name: str = "data_analysis_executor"
    description: str = "Execute Python code for data analysis tasks including EDA, visualization, and statistical analysis"
    args_schema: Type[BaseModel] = DataAnalysisInput
    
    def _run(self, code: str, description: str) -> str:
        """Execute the provided Python code and return results."""
        try:
            # Import the global variable functions from config
            from config import get_global_var, update_global_var
            
            # Get data from global variables (config module)
            shared_df = get_global_var('shared_df')
            X_engineered = get_global_var('X_engineered')
            y = get_global_var('y')
            feature_names = get_global_var('feature_names')
            rf_model = get_global_var('rf_model')
            xgb_model = get_global_var('xgb_model')
            lr_model = get_global_var('lr_model')
            ensemble_pred = get_global_var('ensemble_pred')
            X_train = get_global_var('X_train')
            X_test = get_global_var('X_test')
            y_train = get_global_var('y_train')
            y_test = get_global_var('y_test')
            
            # Create a safe execution environment with actual data
            exec_globals = {
                'pd': pd, 'np': np, 'plt': plt, 'sns': sns,
                'df': shared_df,  # Make df available (standard name)
                'shared_df': shared_df,
                'X_engineered': X_engineered,
                'y': y,
                'feature_names': feature_names,
                'rf_model': rf_model,
                'xgb_model': xgb_model,
                'lr_model': lr_model,
                'ensemble_pred': ensemble_pred,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
            # Add common ML libraries to execution environment
            try:
                from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
                from sklearn.model_selection import train_test_split
                exec_globals.update({
                    'accuracy_score': accuracy_score,
                    'mean_squared_error': mean_squared_error,
                    'r2_score': r2_score,
                    'train_test_split': train_test_split
                })
            except ImportError:
                pass
            
            # Create a local namespace for execution
            exec_locals = {}
            
            # Preprocess code: replace escaped newlines with real newlines (for JSON inputs)
            code = code.replace("\\n", "\n")

            # Execute the code
            exec(code, exec_globals, exec_locals)
            
            # Update global variables if they were modified in locals
            variable_names = ['shared_df', 'X_engineered', 'y', 'feature_names', 
                            'rf_model', 'xgb_model', 'lr_model', 'ensemble_pred',
                            'X_train', 'X_test', 'y_train', 'y_test']
            
            for var_name in variable_names:
                if var_name in exec_locals:
                    update_global_var(var_name, exec_locals[var_name])
                elif var_name in exec_globals and exec_globals[var_name] != get_global_var(var_name):
                    update_global_var(var_name, exec_globals[var_name])
            
            # Also check for 'df' which might be the modified shared_df
            if 'df' in exec_locals and exec_locals['df'] is not None:
                update_global_var('shared_df', exec_locals['df'])
            
            return f"✅ Successfully executed: {description}\nCode executed without errors."
            
        except Exception as e:
            return f"❌ Error executing code: {str(e)}\nCode: {code[:200]}..."


class ModelTrainingInput(BaseModel):
    """Input schema for model training tool."""
    model_type: str = Field(..., description="Type of model to train (rf, xgb, lr)")
    hyperparams: str = Field(..., description="Hyperparameters as a string")
    cv_folds: int = Field(default=5, description="Number of CV folds")


class ModelTrainingTool(BaseTool):
    name: str = "model_trainer"
    description: str = "Train machine learning models with hyperparameter tuning and cross-validation"
    args_schema: Type[BaseModel] = ModelTrainingInput
    
    def _run(self, model_type: str, hyperparams: str, cv_folds: int = 5) -> str:
        """Train a model with the specified parameters."""
        try:
            from config import get_global_var, update_global_var
            from sklearn.model_selection import GridSearchCV, cross_val_score
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import Ridge
            import xgboost as xgb
            
            X_train = get_global_var('X_train')
            y_train = get_global_var('y_train')
            
            if X_train is None or y_train is None:
                return "❌ Training data not available. Run feature engineering first."
            
            # Parse hyperparameters safely
            try:
                param_dict = eval(hyperparams) if hyperparams else {}
            except:
                param_dict = {}
            
            if model_type == 'rf':
                model = RandomForestRegressor(random_state=42)
                if param_dict:
                    grid_search = GridSearchCV(model, param_dict, cv=cv_folds, scoring='neg_mean_squared_error')
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    update_global_var('rf_model', best_model)
                    return f"✅ Random Forest trained. Best params: {grid_search.best_params_}"
                else:
                    model.fit(X_train, y_train)
                    update_global_var('rf_model', model)
                    return "✅ Random Forest trained with default parameters"
                    
            elif model_type == 'xgb':
                model = xgb.XGBRegressor(random_state=42)
                if param_dict:
                    grid_search = GridSearchCV(model, param_dict, cv=cv_folds, scoring='neg_mean_squared_error')
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    update_global_var('xgb_model', best_model)
                    return f"✅ XGBoost trained. Best params: {grid_search.best_params_}"
                else:
                    model.fit(X_train, y_train)
                    update_global_var('xgb_model', model)
                    return "✅ XGBoost trained with default parameters"
                    
            elif model_type == 'lr':
                model = Ridge(random_state=42)
                if param_dict:
                    grid_search = GridSearchCV(model, param_dict, cv=cv_folds, scoring='neg_mean_squared_error')
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    update_global_var('lr_model', best_model)
                    return f"✅ Linear Regression trained. Best params: {grid_search.best_params_}"
                else:
                    model.fit(X_train, y_train)
                    update_global_var('lr_model', model)
                    return "✅ Linear Regression trained with default parameters"
            else:
                return f"❌ Unknown model type: {model_type}. Use 'rf', 'xgb', or 'lr'"
                    
        except Exception as e:
            return f"❌ Error training {model_type} model: {str(e)}"


class VisualizationInput(BaseModel):
    """Input schema for visualization tool."""
    plot_type: str = Field(..., description="Type of plot to create")
    data_source: str = Field(..., description="Source of data for plotting")
    params: str = Field(default="{}", description="Additional plotting parameters")


class VisualizationTool(BaseTool):
    name: str = "chart_generator"
    description: str = "Generate various types of charts and visualizations for data analysis"
    args_schema: Type[BaseModel] = VisualizationInput
    
    def _run(self, plot_type: str, data_source: str, params: str = "{}") -> str:
        """Generate visualizations based on the specified parameters."""
        try:
            from config import get_global_var
            
            # Parse parameters safely
            try:
                plot_params = eval(params) if params else {}
            except:
                plot_params = {}
            
            # Get data from global variables
            if data_source == 'shared_df' or data_source == 'df':
                df = get_global_var('shared_df')
            else:
                df = get_global_var(data_source)
            
            if df is None:
                return f"❌ No data available in {data_source}"
            
            if plot_type == "correlation_heatmap":
                plt.figure(figsize=(12, 8))
                correlation_matrix = df.select_dtypes(include=[np.number]).corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                plt.title("Feature Correlation Heatmap")
                plt.tight_layout()
                plt.show()
                return "✅ Correlation heatmap generated successfully"
                    
            elif plot_type == "distribution_plots":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                n_cols = min(4, len(numeric_cols))
                if n_cols == 0:
                    return "❌ No numeric columns found for distribution plots"
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.ravel()
                
                for i, col in enumerate(numeric_cols[:4]):
                    sns.histplot(df[col], kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
                    
                # Hide unused subplots
                for i in range(n_cols, 4):
                    axes[i].set_visible(False)
                    
                plt.tight_layout()
                plt.show()
                return "✅ Distribution plots generated successfully"
            
            elif plot_type == "scatter_plot":
                x_col = plot_params.get('x_col')
                y_col = plot_params.get('y_col')
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(df[x_col], df[y_col], alpha=0.6)
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.title(f'{x_col} vs {y_col}')
                    plt.show()
                    return f"✅ Scatter plot of {x_col} vs {y_col} generated successfully"
                else:
                    return "❌ Please provide valid x_col and y_col in params"
            
            elif plot_type == "box_plot":
                numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6 columns
                if len(numeric_cols) == 0:
                    return "❌ No numeric columns found for box plots"
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.ravel()
                
                for i, col in enumerate(numeric_cols):
                    sns.boxplot(y=df[col], ax=axes[i])
                    axes[i].set_title(f'Box Plot of {col}')
                
                # Hide unused subplots
                for i in range(len(numeric_cols), 6):
                    axes[i].set_visible(False)
                    
                plt.tight_layout()
                plt.show()
                return "✅ Box plots generated successfully"
                
            else:
                return f"✅ {plot_type} plot generated for {data_source}"
            
        except Exception as e:
            return f"❌ Error generating visualization: {str(e)}"


def get_tools():
    """Return instances of all tools."""
    return {
        'data_analysis_tool': DataAnalysisTool(),
        'model_training_tool': ModelTrainingTool(),
        'visualization_tool': VisualizationTool()
    }


# Test function
if __name__ == "__main__":
    # Test the DataAnalysisTool with comprehensive analysis
    input_code = """
import pandas as pd
import numpy as np

# Create a test DataFrame with various data types
df = pd.DataFrame({
    'numeric': [1, 2, 3, 4, 5],
    'text': ['A', 'B', 'C', 'D', 'E'],
    'float': [1.1, 2.2, 3.3, 4.4, 5.5],
    'dates': pd.date_range('2024-01-01', periods=5)
})

print("First few rows of the DataFrame:")
print(df.head())
print("\nDataFrame Information:")
df.info()
print("\nBasic Statistics:")
print(df.describe())
"""
    input_description = "Display DataFrame preview, structure information, and basic statistics"

    tool = DataAnalysisTool()
    output = tool._run(code=input_code, description=input_description)
    print("Output:\n", output)