# main_advanced.py - Advanced OpenAI-Powered Data Analysis Framework

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from crewai import Crew, Process
import warnings
import chardet
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

warnings.filterwarnings('ignore')

# Import custom modules
from config import DATA_CONFIG, ANALYSIS_CONFIG, REPORT_CONFIG, get_llm, OPENAI_API_KEY
from agents import create_advanced_data_analysis_agents
from tasks import create_data_analysis_tasks

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedDataAnalyzer:
    """Advanced OpenAI-powered data analysis framework with sophisticated missing value handling"""
    
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.llm = None
        self.analysis_config = {}
        self.setup_environment()
    
    def setup_environment(self) -> None:
        """Setup advanced environment with OpenAI optimizations"""
        logger.info("🚀 Initializing Advanced OpenAI Data Analysis Framework...")
        
        # Validate API key
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY.")
        
        logger.info(f" OpenAI API key validated (...{self.api_key[-4:]})")
        
        # Initialize LLM
        try:
            self.llm = get_llm()
            test_response = self.llm.invoke("Test - respond with 'OK'")
            if "ok" in test_response.lower():
                logger.info("OpenAI connection verified")
            else:
                logger.warning("OpenAI connection test returned unexpected response")
        except Exception as e:
            logger.error(f" OpenAI connection failed: {e}")
            raise
        
        # Setup enhanced plotting
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        
        plt.rcParams.update({
            'figure.figsize': (15, 10),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        # Set analysis configuration
        self.analysis_config = {
            'api_validated': True,
            'model': 'gpt-4o-mini',
            'timestamp': datetime.now().isoformat(),
            'features': {
                'advanced_missing_handling': True,
                'ai_insights': True,
                'interactive_plots': True,
                'automated_feature_engineering': True,
                'statistical_testing': True,
                'outlier_analysis': True
            }
        }
        
        logger.info("✅ Advanced environment setup complete")
    
    def detect_encoding(self, file_path: str) -> Optional[str]:
        """Detect file encoding with high accuracy"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(100000) 
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                logger.info(f"🔍 Encoding: {encoding} (confidence: {confidence:.2f})")
                return encoding if confidence > 0.7 else None
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}")
            return None
    
    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Advanced data loading with comprehensive error handling and AI insights"""
        logger.info(f"📊 Loading data from: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            return None
        
        # File analysis
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"📁 File size: {file_size_mb:.2f} MB")
        
        # Detect encoding
        detected_encoding = self.detect_encoding(file_path)
        
        # Try multiple encoding strategies
        encodings = [detected_encoding] if detected_encoding else []
        encodings.extend(['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-8-sig'])
        
        df = None
        for encoding in encodings:
            if encoding is None:
                continue
                
            try:
                logger.info(f"🔄 Trying encoding: {encoding}")
                
                # Smart file type detection and loading
                if file_path.lower().endswith('.csv'):
                    # Try different separators
                    for sep in [',', ';', '\t', '|']:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep, low_memory=False)
                            if df.shape[1] > 1: 
                                logger.info(f"✅ Loaded with encoding='{encoding}', sep='{sep}'")
                                break
                        except:
                            continue
                    if df is not None and df.shape[1] > 1:
                        break
                        
                elif file_path.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                    logger.info("✅ Excel file loaded successfully")
                    break
                    
                elif file_path.lower().endswith('.json'):
                    df = pd.read_json(file_path, encoding=encoding)
                    logger.info("✅ JSON file loaded successfully")
                    break
                    
                elif file_path.lower().endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                    logger.info("✅ Parquet file loaded successfully")
                    break
                    
            except Exception as e:
                logger.debug(f"Failed with {encoding}: {e}")
                continue
        
        if df is None or df.empty:
            logger.error("❌ Failed to load data with all encoding attempts")
            return None
        
        # Advanced data profiling
        self._profile_dataset(df)
        
        return df
    
    def _profile_dataset(self, df: pd.DataFrame) -> None:
        """Comprehensive dataset profiling with AI insights"""
        logger.info("🔍 Performing advanced dataset profiling...")
        
        # Basic statistics
        logger.info(f"📊 Dataset Profile:")
        logger.info(f"   • Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        logger.info(f"   • Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data quality metrics
        missing_total = df.isnull().sum().sum()
        missing_pct = (missing_total / df.size) * 100
        duplicates = df.duplicated().sum()
        
        logger.info(f"   • Missing values: {missing_total:,} ({missing_pct:.2f}%)")
        logger.info(f"   • Duplicates: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
        
        # Column type analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        
        logger.info(f"   • Numeric: {len(numeric_cols)} columns")
        logger.info(f"   • Categorical: {len(categorical_cols)} columns")
        logger.info(f"   • DateTime: {len(datetime_cols)} columns")
        
        # AI-powered data insights
        if len(df) > 0:
            self._get_ai_dataset_insights(df)
    
    def _get_ai_dataset_insights(self, df: pd.DataFrame) -> None:
        """Get AI-powered insights about the dataset"""
        try:
            # Prepare dataset summary for AI analysis
            summary = {
                'shape': df.shape,
                'dtypes': df.dtypes.value_counts().to_dict(),
                'missing_by_column': df.isnull().sum().head(10).to_dict(),
                'sample_data': df.head(3).to_dict()
            }
            
            prompt = f"""
            Analyze this dataset and provide 3-5 key insights:
            
            Dataset Summary:
            - Shape: {summary['shape']}
            - Data types: {summary['dtypes']}
            - Missing values (top 10): {summary['missing_by_column']}
            
            Sample data (first 3 rows):
            {summary['sample_data']}
            
            Provide insights about:
            1. Data quality issues
            2. Potential analysis opportunities
            3. Recommended preprocessing steps
            4. Notable patterns or anomalies
            
            Keep each insight concise (1-2 sentences).
            """
            
            insights = self.llm.invoke(prompt)
            logger.info("🤖 AI Dataset Insights:")
            logger.info(insights)
            
        except Exception as e:
            logger.warning(f"Could not generate AI insights: {e}")
    
    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Advanced missing value treatment with AI-powered strategy selection.
        
        Strategy Rules:
        - Categorical columns:
          - < 5% missing: Use mode
          - 5-15% missing: Use 'Unknown' category
          - > 15% missing: Flag for review
        
        - Numerical columns:
          - < 5% missing: Use mean/median based on skewness
          - 5-15% missing: Use median or advanced imputation
          - > 15% missing: Flag for review
        """
        logger.info("🔧 Starting advanced missing value treatment...")
        
        df_cleaned = df.copy()
        cleaning_log = []
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        
        for column in df.columns:
            missing_pct = missing_percentages[column]
            
            if missing_pct == 0:
                continue
            
            is_numeric = pd.api.types.is_numeric_dtype(df[column])
            
            log_entry = {
                'column': column,
                'missing_percentage': f"{missing_pct:.2f}%",
                'data_type': 'numeric' if is_numeric else 'categorical',
                'strategy': '',
                'reasoning': ''
            }
            
            if is_numeric:
                if missing_pct < 5:
                    # Check distribution characteristics
                    try:
                        skewness = abs(df[column].skew())
                        if pd.isna(skewness):
                            skewness = 0
                        
                        if skewness > 1:
                            # Skewed distribution - use median
                            fill_value = df[column].median()
                            strategy = 'median (skewed distribution)'
                        else:
                            # Normal-ish distribution - use mean
                            fill_value = df[column].mean()
                            strategy = 'mean (normal distribution)'
                        
                        df_cleaned[column].fillna(fill_value, inplace=True)
                        log_entry['strategy'] = f'Imputed with {strategy}'
                        log_entry['reasoning'] = f'Low missing rate, skewness={skewness:.2f}'
                    
                    except Exception:
                        # Fallback to median
                        df_cleaned[column].fillna(df[column].median(), inplace=True)
                        log_entry['strategy'] = 'Imputed with median (fallback)'
                        log_entry['reasoning'] = 'Statistical calculation failed'
                
                elif missing_pct <= 15:
                    # Moderate missing - use median
                    df_cleaned[column].fillna(df[column].median(), inplace=True)
                    log_entry['strategy'] = 'Imputed with median'
                    log_entry['reasoning'] = 'Moderate missing rate - robust choice'
                
                else:
                    # High missing - flag but still impute
                    df_cleaned[column].fillna(df[column].median(), inplace=True)
                    log_entry['strategy'] = 'HIGH MISSING: Median (temporary)'
                    log_entry['reasoning'] = 'Requires domain expertise or advanced methods'
            
            else:  # Categorical
                if missing_pct < 5:
                    # Low missing - use mode
                    mode_val = df[column].mode()[0] if not df[column].mode().empty else 'Unknown'
                    df_cleaned[column].fillna(mode_val, inplace=True)
                    log_entry['strategy'] = f'Imputed with mode: "{mode_val}"'
                    log_entry['reasoning'] = 'Low missing rate - mode is representative'
                
                elif missing_pct <= 15:
                    # Moderate missing - use Unknown
                    df_cleaned[column].fillna('Unknown', inplace=True)
                    log_entry['strategy'] = 'Imputed with "Unknown"'
                    log_entry['reasoning'] = 'Moderate missing rate - explicit unknown category'
                
                else:
                    # High missing - flag
                    df_cleaned[column].fillna('High_Missing_Flag', inplace=True)
                    log_entry['strategy'] = 'HIGH MISSING: Flagged'
                    log_entry['reasoning'] = 'Consider feature engineering or removal'
            
            cleaning_log.append(log_entry)
        
        # Summary
        before_missing = df.isnull().sum().sum()
        after_missing = df_cleaned.isnull().sum().sum()
        
        logger.info(f"  Missing value treatment completed:")
        logger.info(f"   • Missing before: {before_missing:,}")
        logger.info(f"   • Missing after: {after_missing:,}")
        logger.info(f"   • Columns treated: {len(cleaning_log)}")
        
        return df_cleaned, cleaning_log
    
    def run_analysis(self, file_path: str, workflow_type: str = "advanced") -> Dict[str, Any]:
        """Run comprehensive analysis with OpenAI-powered insights"""
        logger.info(f"🚀 Starting {workflow_type} analysis...")
        
        # Load and validate data
        df = self.load_data(file_path)
        if df is None:
            return {'success': False, 'error': 'Failed to load data'}
        
        original_df = df.copy()
        
        # Advanced missing value handling
        df_cleaned, cleaning_log = self.handle_missing_values(df)
        
        # Create namespace for agents
        namespace = {
            'df': df_cleaned,
            'original_df': original_df,
            'cleaning_log': cleaning_log,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'analysis_config': self.analysis_config
        }
        
        try:
            # Create agents and tasks
            agents = create_advanced_data_analysis_agents(namespace)
            tasks = create_data_analysis_tasks(agents, data_variable_name="df")
            
            # Configure workflow
            workflow_configs = {
                "quick": {
                    "tasks": [tasks['data_profiling'], tasks['data_visualization']],
                    "agents": [agents['data_profiler'], agents['visualization']]
                },
                "standard": {
                    "tasks": [
                        tasks['data_profiling'], 
                        tasks['statistical_analysis'],
                        tasks['data_cleaning'],
                        tasks['data_visualization']
                    ],
                    "agents": [
                        agents['data_profiler'],
                        agents['insight_analyst'], 
                        agents['data_cleaner'],
                        agents['visualization']
                    ]
                },
                "advanced": {
                    "tasks": list(tasks.values()),
                    "agents": list(agents.values())
                }
            }
            
            config = workflow_configs.get(workflow_type, workflow_configs["advanced"])
            
            # Create and run crew
            crew = Crew(
                agents=config["agents"],
                tasks=config["tasks"],
                process=Process.sequential,
                verbose=True,
                max_iter=2,
                memory=False  # Disable memory for stability
            )
            
            result = crew.kickoff()
            
            # Generate comprehensive report
            report = self._generate_advanced_report(original_df, df_cleaned, result, cleaning_log, workflow_type)
            
            return {
                'success': True,
                'result': result,
                'report': report,
                'cleaned_data': df_cleaned,
                'cleaning_log': cleaning_log,
                'namespace': namespace
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_advanced_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                                result: Any, cleaning_log: List[Dict], workflow_type: str) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        logger.info("📋 Generating advanced analysis report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate cleaning statistics
        cleaning_stats = {
            'original_shape': original_df.shape,
            'cleaned_shape': cleaned_df.shape,
            'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
            'columns_processed': len(cleaning_log),
            'missing_before': original_df.isnull().sum().sum(),
            'missing_after': cleaned_df.isnull().sum().sum()
        }
        
        # Save cleaned dataset
        cleaned_filename = f"cleaned_dataset_{workflow_type}_{timestamp}.csv"
        cleaned_df.to_csv(cleaned_filename, index=False)
        
        # Save detailed report
        report_filename = f"advanced_analysis_report_{workflow_type}_{timestamp}.md"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Advanced Data Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Workflow:** {workflow_type}\n")
            f.write(f"**Framework:** OpenAI-Powered CrewAI Analysis\n\n")
            f.write("---\n\n")
            
            # Data Overview
            f.write("## 📊 Dataset Overview\n\n")
            f.write(f"- **Original Shape:** {cleaning_stats['original_shape'][0]:,} rows × {cleaning_stats['original_shape'][1]} columns\n")
            f.write(f"- **Cleaned Shape:** {cleaning_stats['cleaned_shape'][0]:,} rows × {cleaning_stats['cleaned_shape'][1]} columns\n")
            f.write(f"- **Data Quality:** {cleaning_stats['missing_after']} missing values remaining\n\n")
            
            # Missing Value Treatment
            f.write("## 🔧 Missing Value Treatment\n\n")
            f.write(f"**Summary:**\n")
            f.write(f"- Missing values before: {cleaning_stats['missing_before']:,}\n")
            f.write(f"- Missing values after: {cleaning_stats['missing_after']:,}\n")
            f.write(f"- Columns processed: {cleaning_stats['columns_processed']}\n\n")
            
            if cleaning_log:
                f.write("**Detailed Treatment Log:**\n\n")
                for entry in cleaning_log:
                    f.write(f"- **{entry['column']}** ({entry['data_type']}, {entry['missing_percentage']} missing)\n")
                    f.write(f"  - Strategy: {entry['strategy']}\n")
                    f.write(f"  - Reasoning: {entry['reasoning']}\n\n")
            
            # Analysis Results
            f.write("## 📈 Analysis Results\n\n")
            if result and hasattr(result, 'raw'):
                f.write(result.raw)
            
            # Files Generated
            f.write(f"\n## 📁 Generated Files\n\n")
            f.write(f"- **Cleaned Dataset:** `{cleaned_filename}`\n")
            f.write(f"- **Analysis Report:** `{report_filename}`\n")
            f.write(f"- **Analysis Log:** `advanced_analysis.log`\n")
        
        logger.info(f"✅ Advanced report saved: {report_filename}")
        
        return {
            'report_file': report_filename,
            'cleaned_file': cleaned_filename,
            'cleaning_stats': cleaning_stats,
            'timestamp': timestamp
        }

# Convenience functions for backward compatibility and easy usage
def run_quick_analysis(file_path: str = None) -> Dict[str, Any]:
    """Run quick analysis"""
    analyzer = AdvancedDataAnalyzer()
    path = file_path or DATA_CONFIG.get('file_path', 'data.csv')
    return analyzer.run_analysis(path, "quick")

def run_standard_analysis(file_path: str = None) -> Dict[str, Any]:
    """Run standard analysis"""
    analyzer = AdvancedDataAnalyzer()
    path = file_path or DATA_CONFIG.get('file_path', 'data.csv')
    return analyzer.run_analysis(path, "standard")

def run_advanced_analysis(file_path: str = None) -> Dict[str, Any]:
    """Run comprehensive advanced analysis"""
    analyzer = AdvancedDataAnalyzer()
    path = file_path or DATA_CONFIG.get('file_path', 'data.csv')
    return analyzer.run_analysis(path, "advanced")

def interactive_analysis():
    """Interactive analysis with user options"""
    print("🎯 Advanced OpenAI Data Analysis Framework")
    print("=" * 60)
    print("Select analysis type:")
    print("1. Quick Analysis (Fast overview)")
    print("2. Standard Analysis (Comprehensive)")
    print("3. Advanced Analysis (Full suite)")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (0-3): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                result = run_quick_analysis()
                print(f"Quick analysis: {'completed' if result['success'] else 'failed'}")
                if not result['success']:
                    print(f"Error: {result.get('error', 'Unknown error')}")
            elif choice == '2':
                result = run_standard_analysis()
                print(f"✅ Standard analysis: {'completed' if result['success'] else 'failed'}")
                if not result['success']:
                    print(f"Error: {result.get('error', 'Unknown error')}")
            elif choice == '3':
                result = run_advanced_analysis()
                print(f"✅ Advanced analysis: {'completed' if result['success'] else 'failed'}")
                if not result['success']:
                    print(f" Error: {result.get('error', 'Unknown error')}")
            else:
                print("❌ Invalid choice. Please enter 0-3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main execution function"""
    print("Advanced OpenAI Data Analysis Framework")
    print("=" * 60)
    
    try:
        # Run default advanced analysis
        result = run_advanced_analysis()
        
        if result['success']:
            print("Analysis completed successfully!")
            print(f" Report: {result['report']['report_file']}")
            print(f"🧹 Cleaned data: {result['report']['cleaned_file']}")
            
            # Print missing value treatment summary
            cleaning_log = result.get('cleaning_log', [])
            if cleaning_log:
                print(f"\n🔧 Missing Value Treatment Summary:")
                for entry in cleaning_log:
                    print(f"  • {entry['column']}: {entry['strategy']}")
        else:
            print(f" Analysis failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f" Execution failed: {e}")

if __name__ == "__main__":
    # Choose execution mode
    print(" Advanced OpenAI Data Analysis Framework")
    print("=" * 60)
    print("Execution options:")
    print("1. Run main analysis")
    print("2. Interactive mode")
    
    try:
        choice = input("Enter choice (1-2, or Enter for main): ").strip()
        
        if choice == '2':
            interactive_analysis()
        else:
            main()
            
    except KeyboardInterrupt:
        print("\n Goodbye")
    except Exception as e:
        print(f" Error: {e}")

# Export key functions
__all__ = [
    'AdvancedDataAnalyzer',
    'run_quick_analysis', 
    'run_standard_analysis', 
    'run_advanced_analysis',
    'interactive_analysis',
    'main'
]
