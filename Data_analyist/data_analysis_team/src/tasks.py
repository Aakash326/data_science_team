# tasks.py

from crewai import Task
from config import DATA_CONFIG

def create_data_analysis_tasks(agents, data_variable_name="df"):
    """Create comprehensive data analysis tasks for the CrewAI agents"""
    
    # Task 1: Data Profiling and Structure Analysis
    data_profiling_task = Task(
        description=(
            f"Perform comprehensive data profiling and structure analysis on the dataset '{data_variable_name}'. "
            f"Your analysis must include:\n\n"
            f"1. **Basic Dataset Information**:\n"
            f"   - Dataset shape (rows × columns)\n"
            f"   - Memory usage and size metrics\n"
            f"   - Column names and their order\n\n"
            f"2. **Data Type Analysis**:\n"
            f"   - Detailed breakdown of each column's data type\n"
            f"   - Identification of numerical vs categorical columns\n"
            f"   - Detection of potential data type inconsistencies\n"
            f"   - Summary of data type distribution across columns\n\n"
            f"3. **Data Quality Assessment**:\n"
            f"   - Missing value analysis (count and percentage per column)\n"
            f"   - Duplicate row detection and analysis\n"
            f"   - Unique value counts for each column\n"
            f"   - Identification of constant or near-constant columns\n\n"
            f"4. **Initial Statistical Overview**:\n"
            f"   - Basic descriptive statistics for numerical columns\n"
            f"   - Value distribution summary for categorical columns\n"
            f"   - Range and variance analysis for numerical data\n\n"
            f"Use the DataProfiler tool to generate this comprehensive profile. "
            f"If additional analysis is needed, use the NotebookCodeExecutor tool to write and execute "
            f"custom Python code for deeper investigation of specific data characteristics."
        ),
        expected_output=(
            "A comprehensive data profiling report containing:\n"
            "- Complete dataset structure summary with dimensions and memory usage\n"
            "- Detailed column-by-column analysis including data types, unique values, and missing data\n"
            "- Data quality assessment highlighting any issues or inconsistencies\n"
            "- Statistical summary of numerical columns with key metrics\n"
            "- Categorical column analysis with value distribution insights\n"
            "- Clear identification of potential data quality issues requiring attention\n"
            "- Professional formatting with clear sections and bullet points for easy reference\n"
            "- Specific recommendations for data cleaning or preprocessing if needed"
        ),
        agent=agents['data_profiler'],
        tools=[agents['data_profiler'].tools[0], agents['data_profiler'].tools[1]]  # DataProfiler, NotebookExecutor
    )
    
    # Task 2: Statistical Analysis and Insights Extraction
    statistical_analysis_task = Task(
        description=(
            f"Conduct comprehensive statistical analysis and extract meaningful insights from '{data_variable_name}'. "
            f"Focus on identifying patterns, trends, relationships, and anomalies. Your analysis should include:\n\n"
            f"1. **Descriptive Statistics Deep Dive**:\n"
            f"   - Advanced descriptive statistics (mean, median, mode, std, variance, skewness, kurtosis)\n"
            f"   - Distribution analysis and normality testing for numerical columns\n"
            f"   - Quartile analysis and percentile distributions\n"
            f"   - Coefficient of variation and relative spread metrics\n\n"
            f"2. **Correlation and Relationship Analysis**:\n"
            f"   - Comprehensive correlation matrix for numerical variables\n"
            f"   - Identification of strong positive and negative correlations (|r| > 0.7)\n"
            f"   - Cross-tabulation analysis for categorical variables\n"
            f"   - Detection of potential multicollinearity issues\n\n"
            f"3. **Pattern and Trend Detection**:\n"
            f"   - Time-based trends if temporal data exists\n"
            f"   - Seasonal patterns or cyclical behavior\n"
            f"   - Categorical distribution patterns and dominant categories\n"
            f"   - Identification of unusual patterns or discontinuities\n\n"
            f"4. **Advanced Insights Extraction**:\n"
            f"   - Business-relevant insights and their implications\n"
            f"   - Anomaly detection and explanation of unusual values\n"
            f"   - Relationship strength assessment and practical significance\n"
            f"   - Recommendations for further investigation or modeling\n\n"
            f"Use InsightExtractor, StatisticsTool, and NotebookCodeExecutor tools to perform this analysis."
        ),
        expected_output=(
            "A detailed statistical analysis report featuring:\n"
            "- Comprehensive descriptive statistics with interpretation of key metrics\n"
            "- Correlation analysis with clear identification of significant relationships\n"
            "- Pattern and trend analysis with business context and implications\n"
            "- List of top 10 most important insights ranked by business relevance\n"
            "- Statistical significance assessments and confidence intervals where applicable\n"
            "- Clear explanation of any anomalies or unusual patterns discovered\n"
            "- Actionable recommendations based on statistical findings\n"
            "- Professional presentation with proper statistical terminology and clear interpretations\n"
            "- Specific insights about data distributions, relationships, and business implications"
        ),
        agent=agents['insight_analyst'],
        tools=[agents['insight_analyst'].tools[0], agents['insight_analyst'].tools[1], agents['insight_analyst'].tools[2]]
    )
    
    # Task 3: Data Visualization and Visual Analysis
    data_visualization_task = Task(
        description=(
            f"Create comprehensive data visualizations for '{data_variable_name}' that effectively communicate "
            f"key insights and patterns. Generate a variety of appropriate charts and graphs:\n\n"
            f"1. **Distribution Visualizations**:\n"
            f"   - Histograms for all numerical columns to show distribution shapes\n"
            f"   - Box plots to identify outliers and quartile distributions\n"
            f"   - Bar charts for categorical variables showing frequency distributions\n"
            f"   - Density plots for continuous variables to show probability distributions\n\n"
            f"2. **Relationship Visualizations**:\n"
            f"   - Correlation heatmap for numerical variables with proper color coding\n"
            f"   - Scatter plots for key variable pairs showing relationships\n"
            f"   - Pair plots for multiple variable relationships (if feasible)\n"
            f"   - Cross-tabulation heatmaps for categorical relationships\n\n"
            f"3. **Outlier and Anomaly Visualizations**:\n"
            f"   - Box plots highlighting outliers with clear identification\n"
            f"   - Scatter plots with outliers marked in different colors\n"
            f"   - Distribution plots showing normal vs anomalous data points\n\n"
            f"4. **Time Series Visualizations** (if applicable):\n"
            f"   - Line plots showing trends over time\n"
            f"   - Seasonal decomposition plots if patterns exist\n"
            f"   - Moving averages and trend analysis\n\n"
            f"5. **Advanced Visualizations**:\n"
            f"   - Multi-dimensional plots for complex relationships\n"
            f"   - Grouped visualizations for categorical breakdowns\n"
            f"   - Statistical summary visualizations\n\n"
            f"Each visualization should include proper titles, axis labels, legends, and annotations. "
            f"Use appropriate color schemes and ensure all charts are publication-ready."
        ),
        expected_output=(
            "A comprehensive collection of data visualizations including:\n"
            "- Distribution plots (histograms, box plots) for all relevant numerical columns\n"
            "- Categorical distribution charts (bar plots) with clear frequency information\n"
            "- Correlation heatmap with annotated correlation coefficients\n"
            "- Key scatter plots showing important variable relationships\n"
            "- Outlier identification plots with clear marking of anomalous points\n"
            "- Time series plots if temporal data is present\n"
            "- All visualizations properly formatted with titles, labels, and legends\n"
            "- Written interpretation accompanying each visualization explaining key insights\n"
            "- Summary of the most important visual findings\n"
            "- Recommendations for stakeholders based on visual analysis\n"
            "- Professional-quality plots suitable for presentations and reports"
        ),
        agent=agents['visualization'],
        tools=[agents['visualization'].tools[0], agents['visualization'].tools[1], agents['visualization'].tools[2]]
    )
    
    # Task 4: Data Cleaning and Quality Improvement
    data_cleaning_task = Task(
        description=(
            f"Perform comprehensive data cleaning and quality improvement on '{data_variable_name}'. "
            f"Focus on identifying and resolving data quality issues while preserving data integrity:\n\n"
            f"1. **Missing Value Analysis and Treatment**:\n"
            f"   - Detailed analysis of missing value patterns and distributions\n"
            f"   - Assessment of missing data mechanisms (MCAR, MAR, NMAR)\n"
            f"   - Implementation of appropriate imputation strategies:\n"
            f"     * Numerical columns: median/mean imputation or advanced methods\n"
            f"     * Categorical columns: mode imputation or 'Unknown' category\n"
            f"     * Time series: forward/backward fill or interpolation\n"
            f"   - Documentation of all imputation decisions and their rationale\n\n"
            f"2. **Duplicate Detection and Removal**:\n"
            f"   - Identification of exact and near-duplicate records\n"
            f"   - Analysis of duplicate patterns and potential causes\n"
            f"   - Safe removal of duplicates with impact assessment\n"
            f"   - Preservation of legitimate repeated observations\n\n"
            f"3. **Data Type Optimization and Correction**:\n"
            f"   - Automatic detection and correction of incorrect data types\n"
            f"   - Conversion of string numbers to numerical types\n"
            f"   - Proper datetime parsing and formatting\n"
            f"   - Boolean type optimization for binary variables\n"
            f"   - Category type optimization for string variables with limited unique values\n\n"
            f"4. **Data Consistency and Format Standardization**:\n"
            f"   - Standardization of text fields (case, spacing, special characters)\n"
            f"   - Validation of data ranges and logical constraints\n"
            f"   - Correction of obvious data entry errors\n"
            f"   - Standardization of categorical values and labels\n\n"
            f"5. **Quality Validation and Documentation**:\n"
            f"   - Before/after comparison of data quality metrics\n"
            f"   - Impact assessment of all cleaning operations\n"
            f"   - Documentation of all changes for reproducibility\n"
            f"   - Final data quality report with recommendations\n\n"
            f"Use DataCleaner, DataProfiler, and NotebookCodeExecutor tools to perform this comprehensive cleaning."
        ),
        expected_output=(
            "A complete data cleaning report including:\n"
            "- Detailed before/after comparison showing improvement in data quality\n"
            "- Missing value treatment summary with chosen strategies and their justification\n"
            "- Duplicate removal report with counts and impact analysis\n"
            "- Data type optimization summary showing all conversions performed\n"
            "- Consistency improvements and standardization applied\n"
            "- Final data quality metrics proving the effectiveness of cleaning operations\n"
            "- Comprehensive documentation of all cleaning decisions for reproducibility\n"
            "- Updated dataset statistics showing improved data quality\n"
            "- Recommendations for ongoing data quality maintenance\n"
            "- Clear statement of any data limitations or quality issues that remain\n"
            "- Cleaned dataset ready for analysis and modeling"
        ),
        agent=agents['data_cleaner'],
        tools=[agents['data_cleaner'].tools[0], agents['data_cleaner'].tools[1], agents['data_cleaner'].tools[2]]
    )
    
    # Task 5: Outlier Detection and Analysis
    outlier_analysis_task = Task(
        description=(
            f"Conduct comprehensive outlier detection and analysis on '{data_variable_name}' using multiple "
            f"statistical methods. Provide detailed analysis of anomalies and their potential impact:\n\n"
            f"1. **Multi-Method Outlier Detection**:\n"
            f"   - IQR (Interquartile Range) method for robust outlier detection\n"
            f"   - Z-score method for normally distributed variables\n"
            f"   - Isolation Forest for multivariate outlier detection\n"
            f"   - Comparison of results across different methods\n\n"
            f"2. **Outlier Characterization and Analysis**:\n"
            f"   - Detailed examination of detected outliers\n"
            f"   - Classification of outliers by severity (mild vs extreme)\n"
            f"   - Analysis of outlier patterns and clustering\n"
            f"   - Assessment of outlier impact on statistical measures\n\n"
            f"3. **Root Cause Analysis**:\n"
            f"   - Investigation of potential causes for outliers:\n"
            f"     * Data entry errors vs legitimate extreme values\n"
            f"     * Measurement errors vs natural variation\n"
            f"     * Systematic issues vs random anomalies\n"
            f"   - Business context consideration for outlier interpretation\n\n"
            f"4. **Impact Assessment**:\n"
            f"   - Effect of outliers on mean, standard deviation, and other statistics\n"
            f"   - Influence on correlation coefficients and relationships\n"
            f"   - Potential impact on predictive modeling\n"
            f"   - Sensitivity analysis with and without outliers\n\n"
            f"5. **Treatment Recommendations**:\n"
            f"   - Specific recommendations for each type of outlier\n"
            f"   - Options: removal, transformation, capping, or retention\n"
            f"   - Risk assessment for each treatment approach\n"
            f"   - Guidelines for future outlier monitoring\n\n"
            f"Use OutlierDetector, StatisticsTool, VisualizationGenerator, and NotebookCodeExecutor tools."
        ),
        expected_output=(
            "A comprehensive outlier analysis report containing:\n"
            "- Multi-method outlier detection results with comparison across techniques\n"
            "- Detailed list of all detected outliers with their values and detection methods\n"
            "- Statistical impact analysis showing how outliers affect key metrics\n"
            "- Root cause analysis distinguishing between errors and legitimate extreme values\n"
            "- Visual representations of outliers through box plots and scatter plots\n"
            "- Classification of outliers by severity and type\n"
            "- Specific treatment recommendations for each category of outliers\n"
            "- Business impact assessment and implications for decision-making\n"
            "- Sensitivity analysis comparing statistics with and without outliers\n"
            "- Guidelines for ongoing outlier monitoring and management\n"
            "- Clear documentation of methodology and assumptions used"
        ),
        agent=agents['outlier_analysis'],
        tools=[agents['outlier_analysis'].tools[0], agents['outlier_analysis'].tools[1], 
               agents['outlier_analysis'].tools[2], agents['outlier_analysis'].tools[3]]
    )
    
    # Task 6: Advanced Statistical Analysis
    advanced_statistics_task = Task(
        description=(
            f"Perform advanced statistical analysis on '{data_variable_name}' with focus on rigorous "
            f"quantitative methods and statistical inference:\n\n"
            f"1. **Advanced Descriptive Statistics**:\n"
            f"   - Complete statistical summary including all moments (mean, variance, skewness, kurtosis)\n"
            f"   - Robust statistics (median, MAD, trimmed means) for comparison\n"
            f"   - Distribution shape analysis and goodness-of-fit tests\n"
            f"   - Confidence intervals for key population parameters\n\n"
            f"2. **Hypothesis Testing and Inference**:\n"
            f"   - Normality tests (Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov)\n"
            f"   - Tests for equal variances and homoscedasticity\n"
            f"   - Comparison tests between groups (if applicable)\n"
            f"   - Multiple testing correction when performing multiple comparisons\n\n"
            f"3. **Correlation and Association Analysis**:\n"
            f"   - Pearson correlation coefficients with significance tests\n"
            f"   - Spearman rank correlation for non-parametric relationships\n"
            f"   - Partial correlation controlling for confounding variables\n"
            f"   - Chi-square tests for categorical variable associations\n\n"
            f"4. **Distribution Analysis and Modeling**:\n"
            f"   - Distribution fitting for continuous variables\n"
            f"   - Parameter estimation with confidence intervals\n"
            f"   - Probability density and cumulative distribution analysis\n"
            f"   - Quantile-quantile plots for distribution comparison\n\n"
            f"5. **Multivariate Statistical Analysis**:\n"
            f"   - Principal component analysis for dimensionality assessment\n"
            f"   - Multivariate normality testing\n"
            f"   - Covariance and precision matrix analysis\n"
            f"   - Statistical power analysis for future study design\n\n"
            f"Use StatisticsTool, InsightExtractor, and NotebookCodeExecutor for comprehensive analysis."
        ),
        expected_output=(
            "An advanced statistical analysis report featuring:\n"
            "- Complete statistical summary with all distributional properties\n"
            "- Hypothesis testing results with p-values and effect sizes\n"
            "- Comprehensive correlation analysis with significance assessments\n"
            "- Distribution fitting results with parameter estimates and goodness-of-fit\n"
            "- Multivariate analysis revealing complex data relationships\n"
            "- Statistical inference conclusions with appropriate confidence levels\n"
            "- Assumptions checking and validation for all statistical tests performed\n"
            "- Clear interpretation of all statistical results in business context\n"
            "- Recommendations for further statistical modeling or analysis\n"
            "- Technical appendix with detailed methodology and assumptions\n"
            "- All results properly formatted with statistical notation and terminology"
        ),
        agent=agents['statistics'],
        tools=[agents['statistics'].tools[0], agents['statistics'].tools[1], agents['statistics'].tools[2]]
    )
    
    # Task 7: Comprehensive EDA Report Generation
    eda_report_task = Task(
        description=(
            f"Synthesize all previous analyses into a comprehensive Exploratory Data Analysis (EDA) report "
            f"for '{data_variable_name}'. Create a professional, executive-ready document that consolidates "
            f"all findings and provides actionable recommendations:\n\n"
            f"1. **Executive Summary**:\n"
            f"   - High-level overview of dataset characteristics and key findings\n"
            f"   - Top 5 most important insights with business implications\n"
            f"   - Critical data quality issues and their resolution status\n"
            f"   - Key recommendations for stakeholders and decision-makers\n\n"
            f"2. **Dataset Overview and Methodology**:\n"
            f"   - Complete dataset description with context and source information\n"
            f"   - Detailed methodology explaining all analysis techniques used\n"
            f"   - Data quality assessment summary with improvement metrics\n"
            f"   - Limitations and assumptions underlying the analysis\n\n"
            f"3. **Detailed Findings by Category**:\n"
            f"   - Data structure and quality analysis results\n"
            f"   - Statistical analysis summary with key metrics\n"
            f"   - Correlation and relationship findings\n"
            f"   - Outlier analysis results and treatment decisions\n"
            f"   - Pattern and trend analysis outcomes\n\n"
            f"4. **Visual Summary**:\n"
            f"   - Key visualizations with clear interpretations\n"
            f"   - Dashboard-style summary charts\n"
            f"   - Before/after comparisons showing data improvements\n"
            f"   - Highlighting of most important visual insights\n\n"
            f"5. **Business Intelligence and Recommendations**:\n"
            f"   - Actionable insights prioritized by business impact\n"
            f"   - Strategic recommendations based on data findings\n"
            f"   - Risk assessments and mitigation strategies\n"
            f"   - Next steps for further analysis or modeling\n\n"
            f"6. **Technical Appendices**:\n"
            f"   - Detailed statistical results and test outcomes\n"
            f"   - Complete data cleaning documentation\n"
            f"   - Methodology details and code references\n"
            f"   - Glossary of technical terms and statistical concepts\n\n"
            f"Use EDAReportGenerator, StatisticsTool, InsightExtractor, and NotebookCodeExecutor tools."
        ),
        expected_output=(
            "A comprehensive, professional EDA report including:\n"
            "- Executive summary suitable for C-level stakeholders\n"
            "- Complete dataset characterization with context and methodology\n"
            "- Consolidated findings from all analysis phases with clear organization\n"
            "- Key insights ranked by business importance and statistical significance\n"
            "- Professional visualizations integrated throughout the report\n"
            "- Detailed statistical appendices for technical stakeholders\n"
            "- Clear, actionable recommendations with implementation priorities\n"
            "- Risk assessment and data limitations clearly documented\n"
            "- Next steps and recommendations for future analysis\n"
            "- Publication-ready formatting with proper citations and references\n"
            "- Complete methodology documentation for reproducibility\n"
            "- Executive dashboard summary for quick reference"
        ),
        agent=agents['eda_report'],
        tools=[agents['eda_report'].tools[0], agents['eda_report'].tools[1], 
               agents['eda_report'].tools[2], agents['eda_report'].tools[3]]
    )
    
    return {
        'data_profiling': data_profiling_task,
        'statistical_analysis': statistical_analysis_task,
        'data_visualization': data_visualization_task,
        'data_cleaning': data_cleaning_task,
        'outlier_analysis': outlier_analysis_task,
        'advanced_statistics': advanced_statistics_task,
        'eda_report': eda_report_task
    }