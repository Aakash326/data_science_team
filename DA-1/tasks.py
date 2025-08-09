from crewai import Task
from agents import (
    data_profiler_agent,
    insight_analyst_agent,
    visualization_agent,
    data_cleaner_agent,
    eda_report_agent,
    statistics_agent,
    outlier_analysis_agent,
)


# Advanced Data Profiling Task
data_profiling_task = Task(
    description=(
        "Perform a deep and structured profiling of the dataset located at '{data_path}' by:\n"
        "1. Identifying dataset dimensions and column names.\n"
        "2. Inferring and validating data types for each column with type confidence.\n"
        "3. Calculating memory usage and optimization opportunities.\n"
        "4. Computing missing value counts and percentages.\n"
        "5. Measuring unique value distribution per column.\n"
        "6. Analyzing statistical properties including skewness and kurtosis.\n"
    ),
    expected_output=(
        "A JSON report containing:\n"
        "- Dataset shape and column metadata.\n"
        "- Column data types with confidence scores and suggestions for correction.\n"
        "- Memory footprint and suggestions to optimize.\n"
        "- Missing value stats and visual representation.\n"
        "- Unique count insights per column.\n"
        "- Skewness and kurtosis stats with flags on irregular distributions."
    ),
    agent=data_profiler_agent,
)

# Data Cleaning Task
data_cleaning_task = Task(
    description=(
        "Perform comprehensive cleaning of the dataset at '{data_path}' by:\n"
        "1. Standardizing column names (lowercase, underscores, trimming).\n"
        "2. Identifying and imputing missing values using column-type-aware strategies.\n"
        "3. Removing exact and fuzzy duplicates.\n"
        "4. Detecting and correcting inconsistent formats (dates, currency, etc.).\n"
        "5. Fixing data type mismatches.\n"
        "6. Logging all cleaning actions for traceability.\n"
    ),
    expected_output=(
        "Output includes:\n"
        "- A cleaned dataset saved as CSV or Parquet.\n"
        "- A cleaning log detailing each fix with row/column references.\n"
        "- Summary of nulls before/after per column.\n"
        "- Number of duplicates removed.\n"
        "- Data type fixes applied.\n"
        "- A separate report highlighting rows that were corrected or flagged."
    ),
    agent=data_cleaner_agent,
)

# EDA Report Task
eda_report_task = Task(
    description=(
        "Create an advanced EDA report for '{data_path}' involving:\n"
        "1. Univariate analysis for all features.\n"
        "2. Bivariate and multivariate visualizations to detect relationships.\n"
        "3. Identification of categorical vs numerical features.\n"
        "4. Visualization of missingness patterns.\n"
        "5. Correlation heatmaps and statistical dependency checks.\n"
        "6. Narrative summaries of key trends and anomalies.\n"
    ),
    expected_output=(
        "Deliverables include:\n"
        "- A comprehensive PDF/HTML EDA report.\n"
        "- Histograms, bar plots, heatmaps, pairplots.\n"
        "- Summary insights on feature relationships.\n"
        "- Visual and textual anomaly flags.\n"
        "- Tables summarizing feature types and cardinality.\n"
        "- Markdown snippet with EDA summary for documentation."
    ),
    agent=eda_report_agent,
)

# Statistics Task
statistics_task = Task(
    description=(
        "Generate robust statistical insights from '{data_path}' including:\n"
        "1. Basic metrics: mean, median, mode, min, max, std.\n"
        "2. Advanced metrics: variance, skewness, kurtosis.\n"
        "3. Correlation matrix for all numerical columns.\n"
        "4. Interquartile ranges and outlier boundary computation.\n"
        "5. Distribution checks and normality tests.\n"
        "6. Highlighting of statistically significant trends.\n"
    ),
    expected_output=(
        "A structured JSON or Markdown output with:\n"
        "- Summary table of all statistical measures.\n"
        "- Correlation heatmap or table.\n"
        "- Highlighted extreme values or skewed features.\n"
        "- Normality check results per column.\n"
        "- Outlier boundaries based on IQR.\n"
        "- Descriptive commentary on metric significance."
    ),
    agent=statistics_agent,
)

# Outlier Analysis Task
outlier_analysis_task = Task(
    description=(
        "Perform a multi-method outlier analysis for dataset '{data_path}':\n"
        "1. Apply IQR and Z-score to detect univariate outliers.\n"
        "2. Use DBSCAN or Isolation Forest for multivariate outlier detection.\n"
        "3. Visualize outliers using box plots and scatter matrices.\n"
        "4. Quantify impact of outliers on statistical measures.\n"
        "5. Suggest appropriate handling strategies (capping/removal).\n"
        "6. Log reproducible outlier tagging logic.\n"
    ),
    expected_output=(
        "Output will include:\n"
        "- List of outlier indices per detection method.\n"
        "- Summary of affected columns and counts.\n"
        "- Plots showing outlier distributions.\n"
        "- Decision recommendations for handling.\n"
        "- Python code snippet for re-identifying outliers.\n"
        "- Flags for potential false positives."
    ),
    agent=outlier_analysis_agent,
)
