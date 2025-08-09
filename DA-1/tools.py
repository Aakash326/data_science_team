import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Union
from crewai.tools import BaseTool
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from io import StringIO
import os
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProfiler(BaseTool):
    """Analyzes column types, missing values, distributions, etc."""
    name = "data_profiler"
    description = "Analyzes dataset structure including column types, missing values, and distributions"

    def run(self, data_path: str) -> Dict:
        try:
            logger.info("Profiling data...")
            # Load the data
            if isinstance(data_path, str) and os.path.exists(data_path):
                df = pd.read_csv(data_path)
            elif isinstance(data_path, pd.DataFrame):
                df = data_path
            else:
                return {"status": "error", "message": f"Invalid data input: {type(data_path)}"}
            
            # Get basic information
            profile = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
                "memory_usage": df.memory_usage(deep=True).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
                "unique_values": {col: df[col].nunique() for col in df.columns},
                "sample_data": df.head(5).to_dict()
            }
            
            # Generate descriptive statistics
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                profile["numeric_stats"] = df[numeric_cols].describe().to_dict()
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                profile["categorical_stats"] = {
                    col: df[col].value_counts().head(10).to_dict() for col in categorical_cols
                }
            
            return {"status": "success", "profile": profile}
        except Exception as e:
            logger.error(f"Error profiling data: {e}")
            return {"status": "error", "message": str(e)}

class NotebookCodeExecutor(BaseTool):
    """Executes Python code (e.g., for pandas, matplotlib, seaborn)."""
    name = "notebook_code_executor"
    description = "Executes Python code for data analysis and visualization"

    def run(self, code: str, data_path: Optional[str] = None) -> Dict:
        try:
            logger.info("Executing code...")
            # Create a local namespace for execution
            local_namespace = {}
            
            # If data_path is provided, load the data into the namespace
            if data_path and os.path.exists(data_path):
                local_namespace['df'] = pd.read_csv(data_path)
            
            # Add common libraries to the namespace
            local_namespace['pd'] = pd
            local_namespace['np'] = np
            local_namespace['plt'] = plt
            local_namespace['sns'] = sns
            
            # Capture output
            output_buffer = StringIO()
            
            # Execute the code
            exec(code, globals(), local_namespace)
            
            # Save any generated plots
            if plt.get_fignums():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = f"plot_{timestamp}.png"
                plt.savefig(plot_path)
                plt.close()
                return {"status": "success", "output": output_buffer.getvalue(), "plot_path": plot_path}
            
            return {"status": "success", "output": output_buffer.getvalue(), "namespace": {k: str(v) for k, v in local_namespace.items() if k not in ['pd', 'np', 'plt', 'sns']}}
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return {"status": "error", "message": str(e)}

class VisualizationGenerator(BaseTool):
    """Creates charts (histograms, boxplots, scatter plots, etc.)."""
    name = "visualization_generator"
    description = "Creates various types of data visualizations"

    def run(self, data_path: str, chart_type: str, x_column: Optional[str] = None, 
            y_column: Optional[str] = None, hue_column: Optional[str] = None, 
            title: Optional[str] = None, figsize: Optional[List[int]] = None) -> Dict:
        try:
            logger.info(f"Generating {chart_type} visualization...")
            
            # Load the data
            if isinstance(data_path, str) and os.path.exists(data_path):
                df = pd.read_csv(data_path)
            elif isinstance(data_path, pd.DataFrame):
                df = data_path
            else:
                return {"status": "error", "message": f"Invalid data input: {type(data_path)}"}
            
            # Set default figure size
            if not figsize:
                figsize = (10, 6)
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Generate visualization based on chart type
            chart_path = f"chart_{chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            if chart_type == "histogram":
                if not x_column:
                    return {"status": "error", "message": "x_column is required for histogram"}
                sns.histplot(data=df, x=x_column, hue=hue_column)
                
            elif chart_type == "boxplot":
                if not x_column:
                    return {"status": "error", "message": "x_column is required for boxplot"}
                sns.boxplot(data=df, x=x_column, y=y_column, hue=hue_column)
                
            elif chart_type == "scatter":
                if not x_column or not y_column:
                    return {"status": "error", "message": "x_column and y_column are required for scatter plot"}
                sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column)
                
            elif chart_type == "pairplot":
                if not x_column:
                    # If no columns specified, use all numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if len(numeric_cols) > 5:  # Limit to 5 columns to avoid huge plots
                        numeric_cols = numeric_cols[:5]
                    g = sns.pairplot(df[numeric_cols], hue=hue_column)
                else:
                    columns_to_plot = [x_column]
                    if y_column:
                        columns_to_plot.append(y_column)
                    g = sns.pairplot(df[columns_to_plot], hue=hue_column)
                plt.tight_layout()
                
            elif chart_type == "heatmap":
                # Create correlation matrix
                numeric_df = df.select_dtypes(include=['number'])
                if numeric_df.shape[1] < 2:
                    return {"status": "error", "message": "Need at least 2 numeric columns for heatmap"}
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
                
            elif chart_type == "bar":
                if not x_column:
                    return {"status": "error", "message": "x_column is required for bar plot"}
                sns.barplot(data=df, x=x_column, y=y_column, hue=hue_column)
                
            elif chart_type == "count":
                if not x_column:
                    return {"status": "error", "message": "x_column is required for count plot"}
                sns.countplot(data=df, x=x_column, hue=hue_column)
                
            else:
                return {"status": "error", "message": f"Unsupported chart type: {chart_type}"}
            
            # Add title if provided
            if title:
                plt.title(title)
                
            # Save the figure
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()
            
            return {"status": "success", "chart_type": chart_type, "chart_path": chart_path}
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return {"status": "error", "message": str(e)}

class StatisticsTool(BaseTool):
    """Computes statistical metrics: mean, median, std, correlation."""
    name = "statistics_tool"
    description = "Computes and interprets statistical metrics for the dataset"

    def run(self, data_path: str, columns: Optional[List[str]] = None) -> Dict:
        try:
            logger.info("Computing statistics...")
            
            # Load the data
            if isinstance(data_path, str) and os.path.exists(data_path):
                df = pd.read_csv(data_path)
            elif isinstance(data_path, pd.DataFrame):
                df = data_path
            else:
                return {"status": "error", "message": f"Invalid data input: {type(data_path)}"}
            
            # Filter columns if specified
            if columns:
                df = df[columns]
            
            # Get numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.empty:
                return {"status": "error", "message": "No numeric columns found in the dataset"}
            
            # Compute basic statistics
            stats = {
                "descriptive": numeric_df.describe().to_dict(),
                "skewness": numeric_df.skew().to_dict(),
                "kurtosis": numeric_df.kurtosis().to_dict(),
                "correlation": numeric_df.corr().to_dict(),
                "unique_counts": {col: df[col].nunique() for col in df.columns}
            }
            
            # Compute additional statistics for categorical columns
            categorical_df = df.select_dtypes(include=['object', 'category'])
            if not categorical_df.empty:
                stats["categorical"] = {
                    col: {
                        "unique_values": categorical_df[col].nunique(),
                        "mode": categorical_df[col].mode()[0] if not categorical_df[col].mode().empty else None,
                        "top_categories": categorical_df[col].value_counts(normalize=True).head(5).to_dict()
                    } for col in categorical_df.columns
                }
            
            return {"status": "success", "statistics": stats}
        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
            return {"status": "error", "message": str(e)}

class EDAReportGenerator(BaseTool):
    """Generates a full Exploratory Data Analysis (EDA) report."""
    name = "eda_report_generator"
    description = "Generates a comprehensive Exploratory Data Analysis report"

    def __init__(self):
        super().__init__()
        self.data_profiler = DataProfiler()
        self.statistics_tool = StatisticsTool()
        self.visualization_generator = VisualizationGenerator()
        self.outlier_detector = OutlierDetector()

    def run(self, data_path: str, output_dir: Optional[str] = "./eda_reports") -> Dict:
        try:
            logger.info("Generating EDA report...")
            
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Load the data
            if isinstance(data_path, str) and os.path.exists(data_path):
                df = pd.read_csv(data_path)
            elif isinstance(data_path, pd.DataFrame):
                df = data_path
            else:
                return {"status": "error", "message": f"Invalid data input: {type(data_path)}"}
            
            # Generate timestamp for report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"eda_report_{timestamp}.html"
            report_path = os.path.join(output_dir, report_filename)
            
            # Get data profile
            profile_result = self.data_profiler.run(df)
            if profile_result["status"] != "success":
                return profile_result
            
            # Get statistics
            stats_result = self.statistics_tool.run(df)
            if stats_result["status"] != "success":
                return stats_result
            
            # Generate key visualizations
            visualizations = []
            
            # 1. Correlation heatmap for numeric columns
            heatmap_result = self.visualization_generator.run(df, "heatmap", title="Correlation Heatmap")
            if heatmap_result["status"] == "success":
                visualizations.append(heatmap_result["chart_path"])
            
            # 2. Distribution plots for key numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                hist_result = self.visualization_generator.run(df, "histogram", x_column=col, 
                                                            title=f"Distribution of {col}")
                if hist_result["status"] == "success":
                    visualizations.append(hist_result["chart_path"])
            
            # 3. Count plots for key categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in cat_cols[:3]:  # Limit to first 3 categorical columns
                count_result = self.visualization_generator.run(df, "count", x_column=col, 
                                                             title=f"Count of {col}")
                if count_result["status"] == "success":
                    visualizations.append(count_result["chart_path"])
            
            # 4. Detect outliers
            outlier_result = self.outlier_detector.run(df)
            
            # Generate HTML report
            with open(report_path, 'w') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>EDA Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .visualization {{ margin: 20px 0; text-align: center; }}
        .visualization img {{ max-width: 100%; height: auto; }}
        pre {{ background-color: #f8f9fa; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Exploratory Data Analysis Report</h1>
    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>Dataset Overview</h2>
    <p>Rows: {df.shape[0]}, Columns: {df.shape[1]}</p>
    
    <h2>Data Profile</h2>
    <pre>{json.dumps(profile_result['profile'], indent=2)}</pre>
    
    <h2>Statistical Summary</h2>
    <pre>{json.dumps(stats_result['statistics'], indent=2)}</pre>
    
    <h2>Key Visualizations</h2>
""")
                
                # Add visualizations
                for viz_path in visualizations:
                    f.write(f"""    <div class="visualization">
        <img src="{os.path.basename(viz_path)}" alt="Visualization">
        <p>{os.path.basename(viz_path)}</p>
    </div>
""")
                
                # Add outlier information if available
                if outlier_result["status"] == "success" and "outliers" in outlier_result:
                    f.write(f"""    <h2>Outlier Analysis</h2>
    <pre>{json.dumps(outlier_result['outliers'], indent=2)}</pre>
""")
                
                f.write("""</body>
</html>""")
            
            return {
                "status": "success", 
                "report_path": report_path,
                "visualizations": visualizations,
                "profile": profile_result["profile"],
                "statistics": stats_result["statistics"]
            }
        except Exception as e:
            logger.error(f"Error generating EDA report: {e}")
            return {"status": "error", "message": str(e)}

class InsightExtractor(BaseTool):
    """Detects key patterns and insights from the data."""
    name = "insight_extractor"
    description = "Extracts key patterns and insights from the dataset"

    def run(self, data_path: str, focus_columns: Optional[List[str]] = None) -> Dict:
        try:
            logger.info("Extracting insights...")
            
            # Load the data
            if isinstance(data_path, str) and os.path.exists(data_path):
                df = pd.read_csv(data_path)
            elif isinstance(data_path, pd.DataFrame):
                df = data_path
            else:
                return {"status": "error", "message": f"Invalid data input: {type(data_path)}"}
            
            # Filter columns if specified
            if focus_columns:
                df = df[focus_columns]
            
            insights = []
            
            # 1. Check for missing values
            missing_vals = df.isnull().sum()
            missing_cols = missing_vals[missing_vals > 0]
            if not missing_cols.empty:
                insights.append({
                    "type": "missing_values",
                    "description": f"Found {len(missing_cols)} columns with missing values",
                    "details": missing_cols.to_dict()
                })
            
            # 2. Check for high correlation between numeric features
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.shape[1] >= 2:
                corr_matrix = numeric_df.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr = [(i, j, corr_matrix.loc[i, j]) for i in corr_matrix.columns 
                            for j in corr_matrix.columns if i != j and corr_matrix.loc[i, j] > 0.7]
                if high_corr:
                    insights.append({
                        "type": "high_correlation",
                        "description": f"Found {len(high_corr)} pairs of highly correlated features",
                        "details": high_corr
                    })
            
            # 3. Check for skewed distributions in numeric columns
            skewed_cols = []
            for col in numeric_df.columns:
                skewness = numeric_df[col].skew()
                if abs(skewness) > 1:
                    skewed_cols.append((col, skewness))
            
            if skewed_cols:
                insights.append({
                    "type": "skewed_distribution",
                    "description": f"Found {len(skewed_cols)} columns with skewed distributions",
                    "details": dict(skewed_cols)
                })
            
            # 4. Check for imbalanced categorical variables
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            imbalanced_cats = []
            for col in cat_cols:
                value_counts = df[col].value_counts(normalize=True)
                if value_counts.iloc[0] > 0.8:  # If dominant category > 80%
                    imbalanced_cats.append((col, value_counts.index[0], value_counts.iloc[0]))
            
            if imbalanced_cats:
                insights.append({
                    "type": "imbalanced_categories",
                    "description": f"Found {len(imbalanced_cats)} categorical columns with imbalanced distribution",
                    "details": imbalanced_cats
                })
            
            # 5. Check for potential outliers in numeric columns
            potential_outliers = {}
            for col in numeric_df.columns:
                q1 = numeric_df[col].quantile(0.25)
                q3 = numeric_df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers_count = ((numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    potential_outliers[col] = outliers_count
            
            if potential_outliers:
                insights.append({
                    "type": "potential_outliers",
                    "description": f"Found potential outliers in {len(potential_outliers)} columns",
                    "details": potential_outliers
                })
            
            # 6. Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                insights.append({
                    "type": "duplicates",
                    "description": f"Found {duplicate_count} duplicate rows",
                    "details": {"count": int(duplicate_count), "percentage": float(duplicate_count / len(df) * 100)}
                })
            
            return {"status": "success", "insights": insights}
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return {"status": "error", "message": str(e)}

class OutlierDetector(BaseTool):
    """Identifies and reports outliers and anomalies."""
    name = "outlier_detector"
    description = "Detects outliers and anomalies in the dataset"

    def run(self, data_path: str, method: str = "iqr", columns: Optional[List[str]] = None) -> Dict:
        try:
            logger.info("Detecting outliers...")
            
            # Load the data
            if isinstance(data_path, str) and os.path.exists(data_path):
                df = pd.read_csv(data_path)
            elif isinstance(data_path, pd.DataFrame):
                df = data_path
            else:
                return {"status": "error", "message": f"Invalid data input: {type(data_path)}"}
            
            # Get numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.empty:
                return {"status": "error", "message": "No numeric columns found in the dataset"}
            
            # Filter columns if specified
            if columns:
                numeric_df = numeric_df[[col for col in columns if col in numeric_df.columns]]
                if numeric_df.empty:
                    return {"status": "error", "message": "No valid numeric columns found in the specified columns"}
            
            outliers = {}
            
            if method.lower() == "iqr":
                # IQR method
                for col in numeric_df.columns:
                    q1 = numeric_df[col].quantile(0.25)
                    q3 = numeric_df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outlier_indices = numeric_df[(numeric_df[col] < lower_bound) | 
                                               (numeric_df[col] > upper_bound)].index.tolist()
                    
                    if outlier_indices:
                        outliers[col] = {
                            "count": len(outlier_indices),
                            "percentage": len(outlier_indices) / len(df) * 100,
                            "indices": outlier_indices[:10],  # Limit to first 10 indices
                            "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                        }
            
            elif method.lower() == "zscore":
                # Z-score method
                for col in numeric_df.columns:
                    z_scores = np.abs((numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std())
                    outlier_indices = np.where(z_scores > 3)[0].tolist()
                    
                    if outlier_indices:
                        outliers[col] = {
                            "count": len(outlier_indices),
                            "percentage": len(outlier_indices) / len(df) * 100,
                            "indices": outlier_indices[:10],  # Limit to first 10 indices
                            "threshold": 3  # Z-score threshold
                        }
            
            elif method.lower() == "isolation_forest":
                # Isolation Forest method
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_df)
                
                model = IsolationForest(contamination=0.05, random_state=42)
                preds = model.fit_predict(scaled_data)
                
                # -1 for outliers, 1 for inliers
                outlier_indices = np.where(preds == -1)[0].tolist()
                
                if outlier_indices:
                    outliers["isolation_forest"] = {
                        "count": len(outlier_indices),
                        "percentage": len(outlier_indices) / len(df) * 100,
                        "indices": outlier_indices[:10]  # Limit to first 10 indices
                    }
            
            else:
                return {"status": "error", "message": f"Unsupported outlier detection method: {method}"}
            
            # Generate visualization for outliers
            if outliers:
                # Create boxplots for columns with outliers
                plt.figure(figsize=(12, len(outliers) * 4))
                for i, col in enumerate(outliers.keys()):
                    if col != "isolation_forest":  # Skip for isolation forest
                        plt.subplot(len(outliers), 1, i + 1)
                        sns.boxplot(x=df[col])
                        plt.title(f"Boxplot for {col}")
                        plt.tight_layout()
                
                # Save the figure
                outlier_viz_path = f"outliers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(outlier_viz_path)
                plt.close()
                
                return {"status": "success", "outliers": outliers, "visualization": outlier_viz_path}
            
            return {"status": "success", "outliers": outliers, "message": "No outliers detected"}
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return {"status": "error", "message": str(e)}

class DataCleaner(BaseTool):
    """Cleans data: handles nulls, removes duplicates, fixes formats."""
    name = "data_cleaner"
    description = "Cleans data by handling missing values, removing duplicates, and fixing formats"

    def run(self, data_path: str, strategies: Optional[Dict] = None, output_path: Optional[str] = None) -> Dict:
        try:
            logger.info("Cleaning data...")
            
            # Load the data
            if isinstance(data_path, str) and os.path.exists(data_path):
                df = pd.read_csv(data_path)
                original_df = df.copy()  # Keep a copy of the original data
            elif isinstance(data_path, pd.DataFrame):
                df = data_path.copy()
                original_df = data_path.copy()
            else:
                return {"status": "error", "message": f"Invalid data input: {type(data_path)}"}
            
            cleaning_report = {
                "original_shape": df.shape,
                "actions": []
            }
            
            # 1. Handle missing values
            if strategies and "missing_values" in strategies:
                missing_strategy = strategies["missing_values"]
                for col, strategy in missing_strategy.items():
                    if col in df.columns:
                        missing_count = df[col].isnull().sum()
                        if missing_count > 0:
                            if strategy == "drop":
                                df = df.dropna(subset=[col])
                                cleaning_report["actions"].append({
                                    "action": "drop_missing",
                                    "column": col,
                                    "rows_affected": int(missing_count)
                                })
                            elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                                df[col] = df[col].fillna(df[col].mean())
                                cleaning_report["actions"].append({
                                    "action": "fill_missing_mean",
                                    "column": col,
                                    "rows_affected": int(missing_count)
                                })
                            elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                                df[col] = df[col].fillna(df[col].median())
                                cleaning_report["actions"].append({
                                    "action": "fill_missing_median",
                                    "column": col,
                                    "rows_affected": int(missing_count)
                                })
                            elif strategy == "mode":
                                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
                                cleaning_report["actions"].append({
                                    "action": "fill_missing_mode",
                                    "column": col,
                                    "rows_affected": int(missing_count)
                                })
                            elif isinstance(strategy, (int, float, str)):
                                df[col] = df[col].fillna(strategy)
                                cleaning_report["actions"].append({
                                    "action": "fill_missing_value",
                                    "column": col,
                                    "value": strategy,
                                    "rows_affected": int(missing_count)
                                })
            else:
                # Default strategy: drop rows with any missing values
                missing_count = df.isnull().sum().sum()
                if missing_count > 0:
                    original_shape = df.shape
                    df = df.dropna()
                    cleaning_report["actions"].append({
                        "action": "drop_missing_rows",
                        "rows_affected": int(original_shape[0] - df.shape[0])
                    })
            
            # 2. Remove duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                df = df.drop_duplicates()
                cleaning_report["actions"].append({
                    "action": "remove_duplicates",
                    "rows_affected": int(duplicate_count)
                })
            
            # 3. Fix data types
            if strategies and "data_types" in strategies:
                for col, dtype in strategies["data_types"].items():
                    if col in df.columns:
                        try:
                            df[col] = df[col].astype(dtype)
                            cleaning_report["actions"].append({
                                "action": "convert_type",
                                "column": col,
                                "new_type": dtype
                            })
                        except Exception as e:
                            cleaning_report["actions"].append({
                                "action": "convert_type_failed",
                                "column": col,
                                "error": str(e)
                            })
            
            # 4. Handle outliers if specified
            if strategies and "outliers" in strategies:
                for col, strategy in strategies["outliers"].items():
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                        outlier_count = outlier_mask.sum()
                        
                        if outlier_count > 0:
                            if strategy == "remove":
                                df = df[~outlier_mask]
                                cleaning_report["actions"].append({
                                    "action": "remove_outliers",
                                    "column": col,
                                    "rows_affected": int(outlier_count)
                                })
                            elif strategy == "cap":
                                df.loc[df[col] < lower_bound, col] = lower_bound
                                df.loc[df[col] > upper_bound, col] = upper_bound
                                cleaning_report["actions"].append({
                                    "action": "cap_outliers",
                                    "column": col,
                                    "rows_affected": int(outlier_count)
                                })
            
            # 5. Normalize/scale numeric columns if specified
            if strategies and "normalize" in strategies:
                for col, method in strategies["normalize"].items():
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        if method == "minmax":
                            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                            cleaning_report["actions"].append({
                                "action": "normalize_minmax",
                                "column": col
                            })
                        elif method == "zscore":
                            df[col] = (df[col] - df[col].mean()) / df[col].std()
                            cleaning_report["actions"].append({
                                "action": "normalize_zscore",
                                "column": col
                            })
            
            # Add final shape to report
            cleaning_report["final_shape"] = df.shape
            cleaning_report["rows_removed"] = original_df.shape[0] - df.shape[0]
            cleaning_report["columns_modified"] = len(set([action["column"] for action in cleaning_report["actions"] 
                                                    if "column" in action]))
            
            # Save cleaned data if output path is provided
            if output_path:
                df.to_csv(output_path, index=False)
                cleaning_report["output_path"] = output_path
            
            return {"status": "success", "cleaned_data": df, "report": cleaning_report}
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return {"status": "error", "message": str(e)}
