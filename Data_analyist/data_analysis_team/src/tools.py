# tools.py

import subprocess
import sys
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import redirect_stdout
from typing import List, Optional, Type, Dict, Any, Union
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Schema definitions
class DataProfilerSchema(BaseModel):
    """Schema for DataProfiler tool"""
    data_variable_name: str = Field(description="Name of the pandas DataFrame variable to profile (e.g., 'df', 'data')")

class NotebookCodeExecutorSchema(BaseModel):
    """Schema for NotebookCodeExecutor tool"""
    code: str = Field(description="The Python code to execute")
    required_libraries: Optional[List[str]] = Field(
        default=None,
        description="List of Python libraries to install before execution"
    )

class VisualizationGeneratorSchema(BaseModel):
    """Schema for VisualizationGenerator tool"""
    data_variable_name: str = Field(description="Name of the pandas DataFrame variable")
    chart_type: str = Field(description="Type of chart: histogram, boxplot, scatter, correlation_heatmap, bar, line")
    columns: Optional[List[str]] = Field(default=None, description="Columns to visualize")
    title: Optional[str] = Field(default=None, description="Chart title")

class StatisticsToolSchema(BaseModel):
    """Schema for StatisticsTool"""
    data_variable_name: str = Field(description="Name of the pandas DataFrame variable")
    columns: Optional[List[str]] = Field(default=None, description="Specific columns to analyze")
    statistics_type: str = Field(description="Type: descriptive, correlation, distribution")

class EDAReportGeneratorSchema(BaseModel):
    """Schema for EDAReportGenerator"""
    data_variable_name: str = Field(description="Name of the pandas DataFrame variable")
    report_sections: Optional[List[str]] = Field(
        default=["overview", "missing_values", "distributions", "correlations", "outliers"],
        description="Sections to include in the report"
    )

class InsightExtractorSchema(BaseModel):
    """Schema for InsightExtractor"""
    data_variable_name: str = Field(description="Name of the pandas DataFrame variable")
    analysis_type: str = Field(description="Type: trends, patterns, relationships, anomalies")

class OutlierDetectorSchema(BaseModel):
    """Schema for OutlierDetector"""
    data_variable_name: str = Field(description="Name of the pandas DataFrame variable")
    method: str = Field(description="Method: IQR, zscore, isolation_forest")
    columns: Optional[List[str]] = Field(default=None, description="Columns to analyze for outliers")

class DataCleanerSchema(BaseModel):
    """Schema for DataCleaner"""
    data_variable_name: str = Field(description="Name of the pandas DataFrame variable")
    cleaning_operations: List[str] = Field(description="Operations: handle_missing, remove_duplicates, fix_dtypes")

# Tool implementations
class DataProfiler(BaseTool):
    name: str = "DataProfiler"
    description: str = (
        "Analyzes dataset structure, column types, missing values, basic statistics, and data quality metrics. "
        "Provides comprehensive data profiling including shape, dtypes, missing value analysis, and basic descriptive statistics. "
        "Input: data_variable_name (name of the DataFrame variable in the global namespace)"
    )
    args_schema: Type[BaseModel] = DataProfilerSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace

    def _run(self, data_variable_name: str) -> str:
        try:
            # Get the DataFrame from the namespace
            if data_variable_name not in self._execution_namespace:
                return f"Error: DataFrame '{data_variable_name}' not found in namespace"
            
            df = self._execution_namespace[data_variable_name]
            
            if not isinstance(df, pd.DataFrame):
                return f"Error: '{data_variable_name}' is not a pandas DataFrame"

            profile_info = []
            profile_info.append("=== DATA PROFILING REPORT ===\n")
            
            # Basic info
            profile_info.append(f"Dataset Shape: {df.shape}")
            profile_info.append(f"Total Rows: {df.shape[0]:,}")
            profile_info.append(f"Total Columns: {df.shape[1]}")
            
            # Column info
            profile_info.append("\n--- COLUMN INFORMATION ---")
            for col in df.columns:
                dtype = str(df[col].dtype)
                unique_count = df[col].nunique()
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                profile_info.append(f"{col}: {dtype} | Unique: {unique_count} | Nulls: {null_count} ({null_pct:.1f}%)")
            
            # Missing values summary
            profile_info.append("\n--- MISSING VALUES SUMMARY ---")
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df)) * 100
            missing_summary = pd.DataFrame({
                'Missing_Count': missing_data,
                'Missing_Percentage': missing_pct
            }).sort_values('Missing_Count', ascending=False)
            
            profile_info.append(missing_summary.to_string())
            
            # Data types summary
            profile_info.append("\n--- DATA TYPES SUMMARY ---")
            dtype_counts = df.dtypes.value_counts()
            profile_info.append(dtype_counts.to_string())
            
            # Numerical columns statistics
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                profile_info.append("\n--- NUMERICAL COLUMNS STATISTICS ---")
                profile_info.append(df[numerical_cols].describe().to_string())
            
            # Categorical columns info
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                profile_info.append("\n--- CATEGORICAL COLUMNS INFO ---")
                for col in categorical_cols:
                    top_values = df[col].value_counts().head(5)
                    profile_info.append(f"\n{col} - Top 5 values:")
                    profile_info.append(top_values.to_string())

            return "\n".join(profile_info)
            
        except Exception as e:
            return f"Error in data profiling: {str(e)}"

class NotebookCodeExecutor(BaseTool):
    name: str = "NotebookCodeExecutor"
    description: str = (
        "Executes Python code directly in the notebook environment with access to global variables. "
        "Can install required libraries and execute pandas, matplotlib, seaborn, sklearn code. "
        "Perfect for data manipulation, analysis, and visualization tasks. "
        "Input: code (Python code string) and optional required_libraries list"
    )
    args_schema: Type[BaseModel] = NotebookCodeExecutorSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace

    def _run(self, code: str, required_libraries: Optional[List[str]] = None) -> str:
        installation_log = ""
        
        # Install required libraries
        if required_libraries:
            installation_log += "--- Installing Libraries ---\n"
            for lib in required_libraries:
                try:
                    process = subprocess.run(
                        [sys.executable, "-m", "pip", "install", lib],
                        capture_output=True, text=True, check=False, timeout=120
                    )
                    if process.returncode == 0:
                        installation_log += f"✓ Successfully installed {lib}\n"
                    else:
                        installation_log += f"✗ Failed to install {lib}: {process.stderr}\n"
                except Exception as e:
                    installation_log += f"✗ Error installing {lib}: {e}\n"
            installation_log += "--- Installation Complete ---\n\n"

        # Execute code
        execution_log = "--- Code Execution ---\n"
        output_buffer = io.StringIO()
        
        try:
            with redirect_stdout(output_buffer):
                exec(code, self._execution_namespace)
            
            execution_output = output_buffer.getvalue()
            execution_log += f"✓ Code executed successfully\n"
            if execution_output:
                execution_log += f"Output:\n{execution_output}\n"
            else:
                execution_log += "No output produced\n"
                
            return installation_log + execution_log
            
        except Exception as e:
            error_output = output_buffer.getvalue()
            execution_log += f"✗ Execution error: {type(e).__name__}: {e}\n"
            if error_output:
                execution_log += f"Partial output: {error_output}\n"
            return installation_log + execution_log

class VisualizationGenerator(BaseTool):
    name: str = "VisualizationGenerator"
    description: str = (
        "Creates various types of data visualizations including histograms, boxplots, scatter plots, "
        "correlation heatmaps, bar charts, and line plots. Automatically handles styling and formatting. "
        "Input: data_variable_name, chart_type, columns (optional), title (optional)"
    )
    args_schema: Type[BaseModel] = VisualizationGeneratorSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace

    def _run(self, data_variable_name: str, chart_type: str, columns: Optional[List[str]] = None, title: Optional[str] = None) -> str:
        try:
            if data_variable_name not in self._execution_namespace:
                return f"Error: DataFrame '{data_variable_name}' not found"
            
            df = self._execution_namespace[data_variable_name]
            
            plt.style.use('default')
            sns.set_palette("husl")
            
            if chart_type == "histogram":
                if columns:
                    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 6*len(columns)))
                    if len(columns) == 1:
                        axes = [axes]
                    for i, col in enumerate(columns):
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                            df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                            axes[i].set_title(f'Distribution of {col}')
                            axes[i].set_xlabel(col)
                            axes[i].set_ylabel('Frequency')
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    axes = axes.ravel()
                    for i, col in enumerate(numeric_cols):
                        df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                        axes[i].set_title(f'Distribution of {col}')
                        
            elif chart_type == "boxplot":
                if columns:
                    numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                plt.figure(figsize=(12, 6))
                df[numeric_cols].boxplot()
                plt.xticks(rotation=45)
                
            elif chart_type == "correlation_heatmap":
                numeric_df = df.select_dtypes(include=[np.number])
                plt.figure(figsize=(10, 8))
                correlation_matrix = numeric_df.corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
                
            elif chart_type == "scatter":
                if columns and len(columns) >= 2:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(df[columns[0]], df[columns[1]], alpha=0.6)
                    plt.xlabel(columns[0])
                    plt.ylabel(columns[1])
                    
            elif chart_type == "bar":
                if columns:
                    col = columns[0]
                    if col in df.columns:
                        plt.figure(figsize=(12, 6))
                        value_counts = df[col].value_counts().head(10)
                        value_counts.plot(kind='bar')
                        plt.xticks(rotation=45)
                        plt.ylabel('Count')
                        
            if title:
                plt.suptitle(title, fontsize=16)
                
            plt.tight_layout()
            plt.show()
            
            return f"✓ {chart_type.replace('_', ' ').title()} visualization created successfully"
            
        except Exception as e:
            return f"Error creating visualization: {str(e)}"

class StatisticsTool(BaseTool):
    name: str = "StatisticsTool"
    description: str = (
        "Computes comprehensive statistical metrics including descriptive statistics, correlation analysis, "
        "and distribution analysis. Provides mean, median, std, skewness, kurtosis, and correlation matrices. "
        "Input: data_variable_name, statistics_type (descriptive/correlation/distribution), columns (optional)"
    )
    args_schema: Type[BaseModel] = StatisticsToolSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace

    def _run(self, data_variable_name: str, statistics_type: str, columns: Optional[List[str]] = None) -> str:
        try:
            if data_variable_name not in self._execution_namespace:
                return f"Error: DataFrame '{data_variable_name}' not found"
            
            df = self._execution_namespace[data_variable_name]
            results = []
            
            if columns:
                df_subset = df[columns]
            else:
                df_subset = df.select_dtypes(include=[np.number])
            
            results.append(f"=== STATISTICAL ANALYSIS ({statistics_type.upper()}) ===\n")
            
            if statistics_type == "descriptive":
                results.append("--- DESCRIPTIVE STATISTICS ---")
                desc_stats = df_subset.describe()
                results.append(desc_stats.to_string())
                
                results.append("\n--- ADDITIONAL STATISTICS ---")
                for col in df_subset.columns:
                    if pd.api.types.is_numeric_dtype(df_subset[col]):
                        skewness = df_subset[col].skew()
                        kurtosis = df_subset[col].kurtosis()
                        results.append(f"{col}:")
                        results.append(f"  Skewness: {skewness:.4f}")
                        results.append(f"  Kurtosis: {kurtosis:.4f}")
                        
            elif statistics_type == "correlation":
                results.append("--- CORRELATION MATRIX ---")
                corr_matrix = df_subset.corr()
                results.append(corr_matrix.to_string())
                
                results.append("\n--- STRONG CORRELATIONS (|r| > 0.7) ---")
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corr.append(f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_val:.4f}")
                
                if strong_corr:
                    results.extend(strong_corr)
                else:
                    results.append("No strong correlations found")
                    
            elif statistics_type == "distribution":
                results.append("--- DISTRIBUTION ANALYSIS ---")
                for col in df_subset.columns:
                    if pd.api.types.is_numeric_dtype(df_subset[col]):
                        # Normality test
                        stat, p_value = stats.normaltest(df_subset[col].dropna())
                        is_normal = p_value > 0.05
                        
                        results.append(f"\n{col}:")
                        results.append(f"  Mean: {df_subset[col].mean():.4f}")
                        results.append(f"  Std: {df_subset[col].std():.4f}")
                        results.append(f"  Normality test p-value: {p_value:.6f}")
                        results.append(f"  Is Normal (p>0.05): {is_normal}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error in statistical analysis: {str(e)}"

class EDAReportGenerator(BaseTool):
    name: str = "EDAReportGenerator"
    description: str = (
        "Generates a comprehensive Exploratory Data Analysis (EDA) report including data overview, "
        "missing values analysis, distribution analysis, correlation analysis, and outlier detection. "
        "Input: data_variable_name, report_sections (optional list of sections to include)"
    )
    args_schema: Type[BaseModel] = EDAReportGeneratorSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace

    def _run(self, data_variable_name: str, report_sections: Optional[List[str]] = None) -> str:
        try:
            if data_variable_name not in self._execution_namespace:
                return f"Error: DataFrame '{data_variable_name}' not found"
            
            df = self._execution_namespace[data_variable_name]
            
            if report_sections is None:
                report_sections = ["overview", "missing_values", "distributions", "correlations", "outliers"]
            
            report = []
            report.append("# COMPREHENSIVE EDA REPORT")
            report.append("=" * 50)
            
            if "overview" in report_sections:
                report.append("\n## 1. DATASET OVERVIEW")
                report.append(f"- Shape: {df.shape}")
                report.append(f"- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                report.append(f"- Columns: {list(df.columns)}")
                
                report.append("\n### Data Types:")
                dtype_summary = df.dtypes.value_counts()
                for dtype, count in dtype_summary.items():
                    report.append(f"- {dtype}: {count} columns")
            
            if "missing_values" in report_sections:
                report.append("\n## 2. MISSING VALUES ANALYSIS")
                missing_data = df.isnull().sum()
                missing_pct = (missing_data / len(df)) * 100
                
                missing_summary = pd.DataFrame({
                    'Missing_Count': missing_data,
                    'Missing_Percentage': missing_pct
                }).sort_values('Missing_Count', ascending=False)
                
                columns_with_missing = missing_summary[missing_summary['Missing_Count'] > 0]
                
                if len(columns_with_missing) > 0:
                    report.append("Columns with missing values:")
                    report.append(columns_with_missing.to_string())
                else:
                    report.append("✓ No missing values found in the dataset")
            
            if "distributions" in report_sections:
                report.append("\n## 3. DISTRIBUTION ANALYSIS")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    report.append(f"\n### {col}:")
                    stats_info = df[col].describe()
                    report.append(f"- Mean: {stats_info['mean']:.4f}")
                    report.append(f"- Std: {stats_info['std']:.4f}")
                    report.append(f"- Skewness: {df[col].skew():.4f}")
                    report.append(f"- Range: [{stats_info['min']:.2f}, {stats_info['max']:.2f}]")
            
            if "correlations" in report_sections:
                report.append("\n## 4. CORRELATION ANALYSIS")
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    
                    # Find strong correlations
                    strong_correlations = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.7:
                                strong_correlations.append(
                                    f"- {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_val:.4f}"
                                )
                    
                    if strong_correlations:
                        report.append("Strong correlations (|r| > 0.7):")
                        report.extend(strong_correlations)
                    else:
                        report.append("No strong correlations found")
                else:
                    report.append("Insufficient numeric columns for correlation analysis")
            
            if "outliers" in report_sections:
                report.append("\n## 5. OUTLIER ANALYSIS")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_pct = (outlier_count / len(df)) * 100
                    
                    report.append(f"\n### {col}:")
                    report.append(f"- Outliers (IQR method): {outlier_count} ({outlier_pct:.2f}%)")
                    if outlier_count > 0:
                        report.append(f"- Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            return "\n".join(report)
            
        except Exception as e:
            return f"Error generating EDA report: {str(e)}"

class InsightExtractor(BaseTool):
    name: str = "InsightExtractor"
    description: str = (
        "Detects and extracts key patterns, trends, relationships, and anomalies from the dataset. "
        "Provides actionable insights about data distribution, correlations, and business implications. "
        "Input: data_variable_name, analysis_type (trends/patterns/relationships/anomalies)"
    )
    args_schema: Type[BaseModel] = InsightExtractorSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace

    def _run(self, data_variable_name: str, analysis_type: str) -> str:
        try:
            if data_variable_name not in self._execution_namespace:
                return f"Error: DataFrame '{data_variable_name}' not found"
            
            df = self._execution_namespace[data_variable_name]
            insights = []
            insights.append(f"=== KEY INSIGHTS ({analysis_type.upper()}) ===\n")
            
            if analysis_type == "trends":
                insights.append("--- TREND ANALYSIS ---")
                
                # Check for time-based trends if date column exists
                date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
                date_like_cols = [col for col in date_cols if 'date' in col.lower() or 'time' in col.lower()]
                
                if date_like_cols:
                    insights.append(f"✓ Potential time series analysis possible with: {date_like_cols}")
                
                # Analyze numerical trends
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    trend_direction = "increasing" if df[col].iloc[-10:].mean() > df[col].iloc[:10].mean() else "decreasing"
                    insights.append(f"- {col}: Shows {trend_direction} trend (comparing first vs last 10 values)")
                    
            elif analysis_type == "patterns":
                insights.append("--- PATTERN ANALYSIS ---")
                
                # Categorical patterns
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    top_category = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                    category_dominance = (df[col].value_counts().iloc[0] / len(df)) * 100
                    insights.append(f"- {col}: Most frequent = '{top_category}' ({category_dominance:.1f}% of data)")
                
                # Numerical patterns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df[col].std() == 0:
                        insights.append(f"- {col}: Constant values detected (no variation)")
                    elif df[col].nunique() < 10:
                        insights.append(f"- {col}: Limited unique values ({df[col].nunique()}) - possible categorical")
                        
            elif analysis_type == "relationships":
                insights.append("--- RELATIONSHIP ANALYSIS ---")
                
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    
                    # Find strongest positive and negative correlations
                    correlations = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if not pd.isna(corr_val):
                                correlations.append({
                                    'pair': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                                    'correlation': corr_val
                                })
                    
                    # Sort by absolute correlation
                    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
                    
                    insights.append("Top relationships by correlation strength:")
                    for i, corr_info in enumerate(correlations[:5]):
                        strength = "Strong" if abs(corr_info['correlation']) > 0.7 else "Moderate" if abs(corr_info['correlation']) > 0.5 else "Weak"
                        direction = "positive" if corr_info['correlation'] > 0 else "negative"
                        insights.append(f"{i+1}. {corr_info['pair']}: {strength} {direction} ({corr_info['correlation']:.3f})")
                        
            elif analysis_type == "anomalies":
                insights.append("--- ANOMALY DETECTION ---")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    # Z-score based anomalies
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    z_anomalies = np.sum(z_scores > 3)
                    z_anomaly_pct = (z_anomalies / len(df)) * 100
                    
                    # IQR based anomalies
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    iqr_anomalies = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])
                    iqr_anomaly_pct = (iqr_anomalies / len(df)) * 100
                    
                    insights.append(f"- {col}:")
                    insights.append(f"  Z-score anomalies (>3σ): {z_anomalies} ({z_anomaly_pct:.2f}%)")
                    insights.append(f"  IQR anomalies: {iqr_anomalies} ({iqr_anomaly_pct:.2f}%)")
                    
                    if z_anomaly_pct > 5 or iqr_anomaly_pct > 5:
                        insights.append(f"  ⚠️  High anomaly rate detected - investigate further")
            
            return "\n".join(insights)
            
        except Exception as e:
            return f"Error extracting insights: {str(e)}"

class OutlierDetector(BaseTool):
    name: str = "OutlierDetector"
    description: str = (
        "Identifies and reports outliers using multiple methods: IQR, Z-score, and Isolation Forest. "
        "Provides detailed outlier analysis with counts, percentages, and outlier values. "
        "Input: data_variable_name, method (IQR/zscore/isolation_forest), columns (optional)"
    )
    args_schema: Type[BaseModel] = OutlierDetectorSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace

    def _run(self, data_variable_name: str, method: str, columns: Optional[List[str]] = None) -> str:
        try:
            if data_variable_name not in self._execution_namespace:
                return f"Error: DataFrame '{data_variable_name}' not found"
            
            df = self._execution_namespace[data_variable_name]
            
            if columns:
                numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            results = []
            results.append(f"=== OUTLIER DETECTION ({method.upper()}) ===\n")
            
            if method.lower() == "iqr":
                results.append("--- IQR METHOD (1.5 * IQR) ---")
                
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_pct = (outlier_count / len(df)) * 100
                    
                    results.append(f"\n{col}:")
                    results.append(f"  Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
                    results.append(f"  Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
                    results.append(f"  Outliers: {outlier_count} ({outlier_pct:.2f}%)")
                    
                    if outlier_count > 0 and outlier_count <= 10:
                        outlier_values = outliers[col].tolist()
                        results.append(f"  Values: {outlier_values}")
                    elif outlier_count > 10:
                        results.append(f"  Sample values: {outliers[col].head(5).tolist()} ... (showing first 5)")
                        
            elif method.lower() == "zscore":
                results.append("--- Z-SCORE METHOD (|z| > 3) ---")
                
                for col in numeric_cols:
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outlier_mask = z_scores > 3
                    outlier_count = np.sum(outlier_mask)
                    outlier_pct = (outlier_count / len(df)) * 100
                    
                    results.append(f"\n{col}:")
                    results.append(f"  Mean: {df[col].mean():.4f}, Std: {df[col].std():.4f}")
                    results.append(f"  Outliers (|z| > 3): {outlier_count} ({outlier_pct:.2f}%)")
                    
                    if outlier_count > 0:
                        outlier_indices = np.where(outlier_mask)[0]
                        outlier_values = df[col].iloc[outlier_indices]
                        
                        if outlier_count <= 10:
                            results.append(f"  Values: {outlier_values.tolist()}")
                        else:
                            results.append(f"  Sample values: {outlier_values.head(5).tolist()} ... (showing first 5)")
                            
            elif method.lower() == "isolation_forest":
                results.append("--- ISOLATION FOREST METHOD ---")
                
                if len(numeric_cols) > 0:
                    # Use all numeric columns for multivariate outlier detection
                    data_for_isolation = df[numeric_cols].dropna()
                    
                    if len(data_for_isolation) > 0:
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outlier_labels = iso_forest.fit_predict(data_for_isolation)
                        
                        outlier_mask = outlier_labels == -1
                        outlier_count = np.sum(outlier_mask)
                        outlier_pct = (outlier_count / len(data_for_isolation)) * 100
                        
                        results.append(f"Data shape for analysis: {data_for_isolation.shape}")
                        results.append(f"Outliers detected: {outlier_count} ({outlier_pct:.2f}%)")
                        
                        if outlier_count > 0:
                            outlier_indices = data_for_isolation.index[outlier_mask]
                            results.append(f"Outlier indices: {outlier_indices.tolist()[:10]}{'...' if len(outlier_indices) > 10 else ''}")
                            
                            # Show statistics for outliers
                            outlier_data = data_for_isolation.loc[outlier_indices]
                            results.append("\nOutlier statistics:")
                            results.append(outlier_data.describe().to_string())
                    else:
                        results.append("No valid data available for Isolation Forest analysis")
                else:
                    results.append("No numeric columns available for Isolation Forest analysis")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error in outlier detection: {str(e)}"

class DataCleaner(BaseTool):
    name: str = "DataCleaner"
    description: str = (
        "Cleans data by handling missing values, removing duplicates, and fixing data types. "
        "Provides before/after statistics and applies multiple cleaning strategies. "
        "Input: data_variable_name, cleaning_operations (list of operations to perform)"
    )
    args_schema: Type[BaseModel] = DataCleanerSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace

    def _run(self, data_variable_name: str, cleaning_operations: List[str]) -> str:
        try:
            if data_variable_name not in self._execution_namespace:
                return f"Error: DataFrame '{data_variable_name}' not found"
            
            df = self._execution_namespace[data_variable_name].copy()
            original_shape = df.shape
            
            results = []
            results.append("=== DATA CLEANING REPORT ===\n")
            results.append(f"Original dataset shape: {original_shape}")
            results.append(f"Original missing values: {df.isnull().sum().sum()}")
            results.append(f"Original duplicates: {df.duplicated().sum()}\n")
            
            if "handle_missing" in cleaning_operations:
                results.append("--- HANDLING MISSING VALUES ---")
                
                missing_before = df.isnull().sum().sum()
                
                # Strategy 1: Drop columns with >50% missing values
                high_missing_cols = df.columns[df.isnull().mean() > 0.5].tolist()
                if high_missing_cols:
                    df = df.drop(columns=high_missing_cols)
                    results.append(f"✓ Dropped columns with >50% missing: {high_missing_cols}")
                
                # Strategy 2: Fill numeric columns with median
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df[col].isnull().any():
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        results.append(f"✓ Filled {col} missing values with median: {median_val:.4f}")
                
                # Strategy 3: Fill categorical columns with mode
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if df[col].isnull().any():
                        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                        df[col].fillna(mode_val, inplace=True)
                        results.append(f"✓ Filled {col} missing values with mode: '{mode_val}'")
                
                missing_after = df.isnull().sum().sum()
                results.append(f"Missing values: {missing_before} → {missing_after}")
            
            if "remove_duplicates" in cleaning_operations:
                results.append("\n--- REMOVING DUPLICATES ---")
                
                duplicates_before = df.duplicated().sum()
                df = df.drop_duplicates()
                duplicates_after = df.duplicated().sum()
                
                results.append(f"✓ Removed {duplicates_before} duplicate rows")
                results.append(f"Duplicates: {duplicates_before} → {duplicates_after}")
            
            if "fix_dtypes" in cleaning_operations:
                results.append("\n--- FIXING DATA TYPES ---")
                
                # Try to convert object columns to appropriate types
                for col in df.select_dtypes(include=['object']).columns:
                    # Try to convert to datetime
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col], errors='ignore')
                            if df[col].dtype != 'object':
                                results.append(f"✓ Converted {col} to datetime")
                        except:
                            pass
                    
                    # Try to convert to numeric
                    else:
                        try:
                            # Check if column contains only numeric values (when not null)
                            non_null_values = df[col].dropna()
                            if len(non_null_values) > 0:
                                # Try to convert a sample
                                sample = non_null_values.head(100)
                                pd.to_numeric(sample, errors='raise')
                                # If successful, convert the whole column
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                results.append(f"✓ Converted {col} to numeric")
                        except:
                            pass
                
                # Convert boolean-like columns
                for col in df.columns:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) == 2 and set(str(v).lower() for v in unique_vals).issubset({'true', 'false', '1', '0', 'yes', 'no'}):
                        df[col] = df[col].map({'True': True, 'False': False, '1': True, '0': False, 
                                             'Yes': True, 'No': False, 'true': True, 'false': False,
                                             'yes': True, 'no': False})
                        results.append(f"✓ Converted {col} to boolean")
            
            # Update the DataFrame in the namespace
            self._execution_namespace[data_variable_name] = df
            
            final_shape = df.shape
            results.append(f"\n=== CLEANING SUMMARY ===")
            results.append(f"Shape: {original_shape} → {final_shape}")
            results.append(f"Rows removed: {original_shape[0] - final_shape[0]}")
            results.append(f"Columns removed: {original_shape[1] - final_shape[1]}")
            results.append(f"Final missing values: {df.isnull().sum().sum()}")
            results.append(f"Final duplicates: {df.duplicated().sum()}")
            results.append(f"✓ Cleaned dataset saved as '{data_variable_name}'")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error in data cleaning: {str(e)}"