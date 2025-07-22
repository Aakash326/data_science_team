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
            # Create a safe execution environment
            exec_globals = {
                'pd': pd, 'np': np, 'plt': plt, 'sns': sns,
                'shared_df': globals().get('shared_df'),
                'X_engineered': globals().get('X_engineered'),
                'y': globals().get('y'),
                'feature_names': globals().get('feature_names'),
                'rf_model': globals().get('rf_model'),
                'xgb_model': globals().get('xgb_model'),
                'lr_model': globals().get('lr_model'),
                'ensemble_pred': globals().get('ensemble_pred'),
                'X_train': globals().get('X_train'),
                'X_test': globals().get('X_test'),
                'y_train': globals().get('y_train'),
                'y_test': globals().get('y_test')
            }
            
            # Execute the code
            exec(code, exec_globals)
            
            # Update global variables if they were modified
            for var_name in ['shared_df', 'X_engineered', 'y', 'feature_names', 
                           'rf_model', 'xgb_model', 'lr_model', 'ensemble_pred',
                           'X_train', 'X_test', 'y_train', 'y_test']:
                if var_name in exec_globals:
                    globals()[var_name] = exec_globals[var_name]
            
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
            from sklearn.model_selection import GridSearchCV, cross_val_score
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import Ridge
            import xgboost as xgb
            
            X_train = globals().get('X_train')
            y_train = globals().get('y_train')
            
            if X_train is None or y_train is None:
                return "❌ Training data not available. Run feature engineering first."
            
            # Parse hyperparameters
            param_dict = eval(hyperparams) if hyperparams else {}
            
            if model_type == 'rf':
                model = RandomForestRegressor(random_state=42)
                if param_dict:
                    grid_search = GridSearchCV(model, param_dict, cv=cv_folds, scoring='neg_mean_squared_error')
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    globals()['rf_model'] = best_model
                    return f"✅ Random Forest trained. Best params: {grid_search.best_params_}"
                else:
                    model.fit(X_train, y_train)
                    globals()['rf_model'] = model
                    return "✅ Random Forest trained with default parameters"
                    
            elif model_type == 'xgb':
                model = xgb.XGBRegressor(random_state=42)
                if param_dict:
                    grid_search = GridSearchCV(model, param_dict, cv=cv_folds, scoring='neg_mean_squared_error')
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    globals()['xgb_model'] = best_model
                    return f"✅ XGBoost trained. Best params: {grid_search.best_params_}"
                else:
                    model.fit(X_train, y_train)
                    globals()['xgb_model'] = model
                    return "✅ XGBoost trained with default parameters"
                    
            elif model_type == 'lr':
                model = Ridge(random_state=42)
                if param_dict:
                    grid_search = GridSearchCV(model, param_dict, cv=cv_folds, scoring='neg_mean_squared_error')
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    globals()['lr_model'] = best_model
                    return f"✅ Linear Regression trained. Best params: {grid_search.best_params_}"
                else:
                    model.fit(X_train, y_train)
                    globals()['lr_model'] = model
                    return "✅ Linear Regression trained with default parameters"
                    
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
            plot_params = eval(params) if params else {}
            
            if plot_type == "correlation_heatmap":
                df = globals().get(data_source)
                if df is not None:
                    plt.figure(figsize=(12, 8))
                    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                    plt.title("Feature Correlation Heatmap")
                    plt.tight_layout()
                    plt.show()
                    return "✅ Correlation heatmap generated successfully"
                    
            elif plot_type == "distribution_plots":
                df = globals().get(data_source)
                if df is not None:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    axes = axes.ravel()
                    for i, col in enumerate(numeric_cols[:4]):
                        sns.histplot(df[col], kde=True, ax=axes[i])
                        axes[i].set_title(f'Distribution of {col}')
                    plt.tight_layout()
                    plt.show()
                    return "✅ Distribution plots generated successfully"
                    
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
if __name__ == "__main__":
    input_code = """
import pandas as pd
import numpy as np
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df.describe()
"""
    input_description = "Perform basic descriptive statistics using pandas."

    tool = DataAnalysisTool()
    output = tool._run(code=input_code, description=input_description)
    print("Output:\n", output)