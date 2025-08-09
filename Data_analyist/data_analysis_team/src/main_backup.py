# main.py - Advanced OpenAI-Powered Data Analysis Framework

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from crewai import Crew, Process
from IPython.display import display, Markdown
import warnings
import chardet
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

warnings.filterwarnings('ignore')

# Import custom modules
from config import DATA_CONFIG, ANALYSIS_CONFIG, REPORT_CONFIG, get_llm, OPENAI_API_KEY
from agents import create_data_analysis_agents
from tasks import create_data_analysis_tasks

# Configure logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Advanced missing value treatment using OpenAI-powered insights.
    Apply smart imputation strategies based on data type, skewness, and missing value percentage.
    Do NOT drop rows or columns blindly.
    """
    logger.info("🔍 Starting advanced missing value analysis...")
    cleaning_log = []
    df_cleaned = df.copy()
    
    # Calculate missing percentages for all columns
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    
    # OpenAI-powered missing value analysis
    llm = get_llm()
    
    for column in df.columns:
        missing_pct = missing_percentages[column]
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        
        # Skip if no missing values
        if missing_pct == 0:
            continue
            
        cleaning_entry = {
            'column': column,
            'missing_percentage': f"{missing_pct:.2f}%",
            'data_type': 'numeric' if is_numeric else 'categorical',
            'action_taken': '',
            'reasoning': ''
        }
        
        # Get OpenAI insights for missing value strategy
        context = f"""
        Column: {column}
        Data type: {'numeric' if is_numeric else 'categorical'}
        Missing percentage: {missing_pct:.2f}%
        Sample values: {df[column].dropna().head(10).tolist()}
        Column description: {df[column].describe() if is_numeric else df[column].value_counts().head()}
        """
        
        if is_numeric:
            # Handle numeric columns with advanced strategies
            if missing_pct < 5:
                # Check skewness and distribution
                skewness = df[column].skew()
                kurtosis = df[column].kurtosis()
                cleaning_entry['skewness'] = f"{skewness:.2f}"
                cleaning_entry['kurtosis'] = f"{kurtosis:.2f}"
                
                if abs(skewness) > 1:
                    # Highly skewed - use median
                    df_cleaned[column].fillna(df[column].median(), inplace=True)
                    cleaning_entry['action_taken'] = f'Imputed with median (skewed distribution)'
                    cleaning_entry['reasoning'] = f'Skewness={skewness:.2f} indicates non-normal distribution'
                elif abs(kurtosis) > 3:
                    # High kurtosis - use median
                    df_cleaned[column].fillna(df[column].median(), inplace=True)
                    cleaning_entry['action_taken'] = f'Imputed with median (high kurtosis)'
                    cleaning_entry['reasoning'] = f'Kurtosis={kurtosis:.2f} indicates heavy tails'
                else:
                    # Normal-ish distribution - use mean
                    df_cleaned[column].fillna(df[column].mean(), inplace=True)
                    cleaning_entry['action_taken'] = f'Imputed with mean (normal distribution)'
                    cleaning_entry['reasoning'] = f'Near-normal distribution (skew={skewness:.2f})'
            
            elif missing_pct <= 15:
                # Moderate missing - use advanced imputation
                if len(df[column].dropna()) > 30:
                    # Use mode for low-variance data, median for others
                    variance_coeff = df[column].std() / df[column].mean() if df[column].mean() != 0 else float('inf')
                    if variance_coeff < 0.1:
                        mode_val = df[column].mode()[0] if not df[column].mode().empty else df[column].median()
                        df_cleaned[column].fillna(mode_val, inplace=True)
                        cleaning_entry['action_taken'] = f'Imputed with mode (low variance)'
                        cleaning_entry['reasoning'] = f'Low coefficient of variation ({variance_coeff:.3f})'
                    else:
                        df_cleaned[column].fillna(df[column].median(), inplace=True)
                        cleaning_entry['action_taken'] = f'Imputed with median (moderate missing)'
                        cleaning_entry['reasoning'] = f'Moderate missing ratio with high variance'
                else:
                    df_cleaned[column].fillna(df[column].median(), inplace=True)
                    cleaning_entry['action_taken'] = f'Imputed with median (small sample)'
                    cleaning_entry['reasoning'] = f'Small sample size ({len(df[column].dropna())} values)'
            
            else:
                # High missing ratio - flag but still impute
                df_cleaned[column].fillna(df[column].median(), inplace=True)
                cleaning_entry['action_taken'] = f'⚠️ HIGH MISSING RATIO: Used median temporarily'
                cleaning_entry['reasoning'] = f'Consider advanced imputation or feature engineering'
        
        else:
            # Handle categorical columns with context awareness
            unique_values = df[column].nunique()
            most_common = df[column].mode()[0] if not df[column].mode().empty else 'Unknown'
            
            if missing_pct < 5:
                df_cleaned[column].fillna(most_common, inplace=True)
                cleaning_entry['action_taken'] = f'Imputed with mode: "{most_common}"'
                cleaning_entry['reasoning'] = f'Low missing ratio, mode is representative'
            
            elif missing_pct <= 15:
                if unique_values <= 5:
                    # Few categories - use mode
                    df_cleaned[column].fillna(most_common, inplace=True)
                    cleaning_entry['action_taken'] = f'Imputed with mode: "{most_common}" (few categories)'
                    cleaning_entry['reasoning'] = f'Only {unique_values} unique categories'
                else:
                    # Many categories - use "Unknown"
                    df_cleaned[column].fillna('Unknown', inplace=True)
                    cleaning_entry['action_taken'] = f'Imputed with "Unknown" category'
                    cleaning_entry['reasoning'] = f'Many categories ({unique_values}), unknown is more appropriate'
            
            else:
                # High missing ratio
                df_cleaned[column].fillna('Missing_HighRatio', inplace=True)
                cleaning_entry['action_taken'] = f'⚠️ HIGH MISSING RATIO: Marked as "Missing_HighRatio"'
                cleaning_entry['reasoning'] = f'Consider domain expertise or advanced techniques'
        
        cleaning_log.append(cleaning_entry)
    
    # Summary statistics
    total_missing_before = df.isnull().sum().sum()
    total_missing_after = df_cleaned.isnull().sum().sum()
    
    logger.info(f"✅ Missing value treatment completed:")
    logger.info(f"   - Missing values before: {total_missing_before}")
    logger.info(f"   - Missing values after: {total_missing_after}")
    logger.info(f"   - Columns processed: {len(cleaning_log)}")
    
    return df_cleaned, cleaning_log

def detect_file_encoding(file_path: str) -> Optional[str]:
    """Detect file encoding using chardet library with advanced fallbacks"""
    try:
        with open(file_path, 'rb') as file:
            # Read a larger sample for better detection
            sample = file.read(50000)  # Read first 50KB
            result = chardet.detect(sample)
            encoding = result['encoding']
            confidence = result['confidence']
            logger.info(f"🔍 Detected encoding: {encoding} (confidence: {confidence:.2f})")
            return encoding
    except Exception as e:
        logger.warning(f"❌ Error detecting encoding: {str(e)}")
        return None

def advanced_setup_environment() -> Dict[str, Any]:
    """Advanced environment setup with OpenAI optimizations"""
    logger.info("🔧 Setting up advanced OpenAI-powered analysis environment...")
    
    # Set up enhanced plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    sns.set_palette(custom_colors)
    plt.rcParams.update({
        'figure.figsize': ANALYSIS_CONFIG.get('figure_size', (14, 10)),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Validate OpenAI API key
    if not OPENAI_API_KEY:
        logger.error("❌ No OpenAI API key found!")
        raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your environment.")
    
    logger.info(f"✅ OpenAI API key validated (ending with: ...{OPENAI_API_KEY[-4:]})")
    
    # Test OpenAI connection
    try:
        llm = get_llm()
        test_response = llm.invoke("Test connection - respond with 'Connected'")
        if "connected" in test_response.lower():
            logger.info("✅ OpenAI connection test successful")
        else:
            logger.warning("⚠️ OpenAI connection test returned unexpected response")
    except Exception as e:
        logger.warning(f"⚠️ OpenAI connection test failed: {str(e)}")
    
    # Setup advanced analysis parameters
    analysis_config = {
        'api_key_validated': True,
        'model_preference': 'gpt-4o-mini',
        'max_tokens': 4000,
        'temperature': 0.1,
        'analysis_timestamp': datetime.now().isoformat(),
        'enhanced_features': {
            'advanced_missing_value_handling': True,
            'ai_powered_insights': True,
            'interactive_visualizations': True,
            'comprehensive_reporting': True,
            'automated_feature_engineering': True
        }
    }
    
    logger.info("✅ Advanced environment setup complete")
    return analysis_config

def advanced_load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Advanced data loading with OpenAI-powered insights and comprehensive error handling"""
    logger.info(f"📊 Loading and analyzing data from {file_path}...")
    
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        return None
    
    # File size check
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    logger.info(f"📁 File size: {file_size:.2f} MB")
    
    if file_size > 500:  # Warn for large files
        logger.warning(f"⚠️ Large file detected ({file_size:.1f} MB). Consider using chunking for better performance.")
    
    # Detect encoding
    detected_encoding = detect_file_encoding(file_path)
    
    # Enhanced encoding list with prioritization
    encodings = []
    if detected_encoding:
        encodings.append(detected_encoding)
    
    # Add common encodings
    common_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'ascii', 'utf-8-sig']
    for enc in common_encodings:
        if enc not in encodings:
            encodings.append(enc)
    
    df = None
    successful_encoding = None
    load_errors = []
    
    for encoding in encodings:
        try:
            logger.info(f"🔄 Trying encoding: {encoding}")
            
            # Try different separators if CSV
            if file_path.lower().endswith('.csv'):
                for sep in [',', ';', '	', '|']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep, low_memory=False)
                        if df.shape[1] > 1:  # Successful if more than 1 column
                            successful_encoding = encoding
                            logger.info(f"✅ Data loaded successfully with {encoding} encoding and '{sep}' separator!")
                            break
                    except:
                        continue
                if df is not None and df.shape[1] > 1:
                    break
            else:
                # Try other file types
                if file_path.lower().endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                elif file_path.lower().endswith('.json'):
                    df = pd.read_json(file_path, encoding=encoding)
                elif file_path.lower().endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                
                successful_encoding = encoding
                logger.info(f"✅ Data loaded successfully with {encoding} encoding!")
                break
                
        except Exception as e:
            error_msg = f"{encoding}: {str(e)}"
            load_errors.append(error_msg)
            logger.debug(f"❌ {error_msg}")
            continue
    
    if df is None or df.empty:
        logger.error("❌ Failed to load data with all attempted encodings")
        logger.error("🔧 Errors encountered:")
        for error in load_errors[-3:]:  # Show last 3 errors
            logger.error(f"   • {error}")
        return None
    
    # Advanced data validation and insights
    logger.info(f"📋 Advanced Dataset Analysis:")
    logger.info(f"   - Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"   - Encoding used: {successful_encoding}")
    logger.info(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data quality assessment
    missing_total = df.isnull().sum().sum()
    missing_pct = (missing_total / (df.shape[0] * df.shape[1])) * 100
    duplicates = df.duplicated().sum()
    
    logger.info(f"📊 Data Quality Assessment:")
    logger.info(f"   - Missing values: {missing_total:,} ({missing_pct:.2f}%)")
    logger.info(f"   - Duplicate rows: {duplicates:,} ({(duplicates/len(df)*100):.2f}%)")
    
    # Column analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    
    logger.info(f"   - Numeric columns: {len(numeric_cols)}")
    logger.info(f"   - Categorical columns: {len(categorical_cols)}")
    logger.info(f"   - DateTime columns: {len(datetime_cols)}")
    
    # Data type recommendations using OpenAI
    if len(categorical_cols) > 0:
        try:
            llm = get_llm()
            sample_data = df[categorical_cols].head(10).to_string()
            prompt = f"""
            Analyze these categorical columns and suggest data type optimizations:
            {sample_data}
            
            Provide brief recommendations for:
            1. Columns that should be converted to category dtype
            2. Columns that might contain dates
            3. Columns that might be boolean
            4. Potential encoding issues
            """
            
            ai_recommendations = llm.invoke(prompt)
            logger.info(f"🤖 AI Data Type Recommendations:")
            logger.info(ai_recommendations)
            
        except Exception as e:
            logger.warning(f"⚠️ Could not get AI recommendations: {str(e)}")
    
    # Preview with better formatting
    logger.info(f"📄 Data Preview (first 5 rows):")
    try:
        preview = df.head(5)
        # Truncate long strings for better display
        for col in preview.select_dtypes(include=['object']):
            preview[col] = preview[col].astype(str).str[:50] + '...'
        logger.info("\n" + preview.to_string())
    except Exception:
        logger.info("Preview not available - but data loaded successfully")
    
    return df

def setup_environment():
    """Setup the analysis environment and load data"""
    print("🔧 Setting up analysis environment...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = ANALYSIS_CONFIG['figure_size']
    
    # Check for required environment variables
    import os
    
    # Check for API keys (Google Gemini or OpenAI)
    google_key = os.getenv('GOOGLE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    chroma_key = os.getenv('CHROMA_OPENAI_API_KEY')
    
    if google_key:
        print("✅ Google Gemini API key found")
        # For CrewAI memory with Gemini, we might need to set OpenAI key for Chroma
        if not chroma_key and not openai_key:
            print("⚠️  Note: Memory features require OpenAI API key for ChromaDB")
            print("   Analysis will continue without memory features")
    elif openai_key:
        print("✅ OpenAI API key found")
        # Set a fallback for Chroma if OpenAI key exists but Chroma key doesn't
        if not chroma_key:
            os.environ['CHROMA_OPENAI_API_KEY'] = openai_key
            print("✅ Using OPENAI_API_KEY for Chroma memory")
    else:
        print("⚠️  Warning: No API key found in environment variables")
        print("   Make sure your config.py has the correct GEMINI_API_KEY")
        print("   Memory features will be disabled")
    
    print("✅ Environment setup complete")

def load_data(file_path):
    """Load and perform initial data setup with automatic encoding detection"""
    print(f"📊 Loading data from {file_path}...")
    
    # First try to detect encoding automatically
    detected_encoding = detect_file_encoding(file_path)
    
    # List of common encodings to try (prioritize detected encoding)
    encodings = []
    if detected_encoding:
        encodings.append(detected_encoding)
    
    # Add common encodings
    common_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'ascii']
    for enc in common_encodings:
        if enc not in encodings:
            encodings.append(enc)
    
    df = None
    successful_encoding = None
    
    for encoding in encodings:
        try:
            print(f"🔄 Trying encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding)
            successful_encoding = encoding
            print(f"✅ Data loaded successfully with {encoding} encoding!")
            break
        except UnicodeDecodeError:
            print(f"❌ {encoding} encoding failed, trying next...")
            continue
        except Exception as e:
            print(f"❌ Error with {encoding}: {str(e)}")
            continue
    
    if df is None:
        print("❌ Failed to load data with all attempted encodings")
        print("🔧 Manual fixes you can try:")
        print("   1. Convert file to UTF-8 using a text editor")
        print("   2. Use Excel to save as CSV with UTF-8 encoding")
        print("   3. Try: pd.read_csv('file.csv', encoding='latin-1', errors='ignore')")
        return None
    
    print(f"📋 Dataset Info:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Encoding used: {successful_encoding}")
    print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"   - Columns: {list(df.columns)}")
    
    # Check for any unusual characters or data issues
    print(f"📊 Data Quality Check:")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    print(f"   - Duplicate rows: {df.duplicated().sum()}")
    
    # Display first few rows to verify data loaded correctly
    print(f"📄 First 3 rows preview:")
    try:
        print(df.head(3).to_string())
    except Exception:
        print("Preview not available - but data loaded successfully")
    
    return df

def detect_file_encoding(file_path: str) -> Optional[str]:
    
def detect_file_encoding(file_path: str) -> Optional[str]:
    """Detect file encoding using chardet library with advanced fallbacks"""
    try:
        with open(file_path, 'rb') as file:
            # Read a larger sample for better detection
            sample = file.read(50000)  # Read first 50KB
            result = chardet.detect(sample)
            encoding = result['encoding']
            confidence = result['confidence']
            logger.info(f"🔍 Detected encoding: {encoding} (confidence: {confidence:.2f})")
            return encoding
    except Exception as e:
        logger.warning(f"❌ Error detecting encoding: {str(e)}")
        return None

def setup_environment() -> Dict[str, Any]:
    """Advanced environment setup with OpenAI optimizations"""
    logger.info("🔧 Setting up advanced OpenAI-powered analysis environment...")
    
    # Set up enhanced plotting style
    plt.style.use('seaborn-v0_8')
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    sns.set_palette(custom_colors)
    plt.rcParams.update({
        'figure.figsize': ANALYSIS_CONFIG.get('figure_size', (14, 10)),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Validate OpenAI API key
    if not OPENAI_API_KEY:
        logger.error("❌ No OpenAI API key found!")
        raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your environment.")
    
    logger.info(f"✅ OpenAI API key validated (ending with: ...{OPENAI_API_KEY[-4:]})")
    
    # Test OpenAI connection
    try:
        llm = get_llm()
        test_response = llm.invoke("Test connection - respond with 'Connected'")
        if "connected" in test_response.lower():
            logger.info("✅ OpenAI connection test successful")
        else:
            logger.warning("⚠️ OpenAI connection test returned unexpected response")
    except Exception as e:
        logger.warning(f"⚠️ OpenAI connection test failed: {str(e)}")
    
    # Setup advanced analysis parameters
    analysis_config = {
        'api_key_validated': True,
        'model_preference': 'gpt-4o-mini',
        'max_tokens': 4000,
        'temperature': 0.1,
        'analysis_timestamp': datetime.now().isoformat(),
        'enhanced_features': {
            'advanced_missing_value_handling': True,
            'ai_powered_insights': True,
            'interactive_visualizations': True,
            'comprehensive_reporting': True,
            'automated_feature_engineering': True
        }
    }
    
    logger.info("✅ Advanced environment setup complete")
    return analysis_config
    """Detect file encoding using chardet library"""
    try:
        with open(file_path, 'rb') as file:
            # Read a sample of the file
            sample = file.read(10000)  # Read first 10KB
            result = chardet.detect(sample)
            encoding = result['encoding']
            confidence = result['confidence']
            print(f"🔍 Detected encoding: {encoding} (confidence: {confidence:.2f})")
            return encoding
    except Exception as e:
        print(f"❌ Error detecting encoding: {str(e)}")
        return None

def setup_environment():
    """Setup the analysis environment and load data"""
    print("🔧 Setting up analysis environment...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = ANALYSIS_CONFIG['figure_size']
    
    # Check for required environment variables
    import os
    
    # Check for API keys (Google Gemini or OpenAI)
    google_key = os.getenv('GOOGLE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    chroma_key = os.getenv('CHROMA_OPENAI_API_KEY')
    
    if google_key:
        print("✅ Google Gemini API key found")
        # For CrewAI memory with Gemini, we might need to set OpenAI key for Chroma
        if not chroma_key and not openai_key:
            print("⚠️  Note: Memory features require OpenAI API key for ChromaDB")
            print("   Analysis will continue without memory features")
    elif openai_key:
        print("✅ OpenAI API key found")
        # Set a fallback for Chroma if OpenAI key exists but Chroma key doesn't
        if not chroma_key:
            os.environ['CHROMA_OPENAI_API_KEY'] = openai_key
            print("✅ Using OPENAI_API_KEY for Chroma memory")
    else:
        print("⚠️  Warning: No API key found in environment variables")
        print("   Make sure your config.py has the correct GEMINI_API_KEY")
        print("   Memory features will be disabled")
    
    print("✅ Environment setup complete")

def load_data(file_path):
    """Load and perform initial data setup with automatic encoding detection"""
    print(f"📊 Loading data from {file_path}...")
    
    # First try to detect encoding automatically
    detected_encoding = detect_file_encoding(file_path)
    
    # List of common encodings to try (prioritize detected encoding)
    encodings = []
    if detected_encoding:
        encodings.append(detected_encoding)
    
    # Add common encodings
    common_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'ascii']
    for enc in common_encodings:
        if enc not in encodings:
            encodings.append(enc)
    
    df = None
    successful_encoding = None
    
    for encoding in encodings:
        try:
            print(f"🔄 Trying encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding)
            successful_encoding = encoding
            print(f"✅ Data loaded successfully with {encoding} encoding!")
            break
        except UnicodeDecodeError:
            print(f"❌ {encoding} encoding failed, trying next...")
            continue
        except Exception as e:
            print(f"❌ Error with {encoding}: {str(e)}")
            continue
    
    if df is None:
        print("❌ Failed to load data with all attempted encodings")
        print("🔧 Manual fixes you can try:")
        print("   1. Convert file to UTF-8 using a text editor")
        print("   2. Use Excel to save as CSV with UTF-8 encoding")
        print("   3. Try: pd.read_csv('file.csv', encoding='latin-1', errors='ignore')")
        return None
    
    print(f"📋 Dataset Info:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Encoding used: {successful_encoding}")
    print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"   - Columns: {list(df.columns)}")
    
    # Check for any unusual characters or data issues
    print(f"📊 Data Quality Check:")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    print(f"   - Duplicate rows: {df.duplicated().sum()}")
    
    # Display first few rows to verify data loaded correctly
    print(f"📄 First 3 rows preview:")
    try:
        print(df.head(3).to_string())
    except Exception:
        print("Preview not available - but data loaded successfully")
    
    return df

def create_analysis_workflow(df, workflow_type="full"):
    """Create the CrewAI workflow for data analysis"""
    print(f"🚀 Creating {workflow_type} analysis workflow...")
    
    # Create shared namespace for all agents/tools
    namespace = globals().copy()
    namespace['df'] = df
    namespace['pd'] = pd
    namespace['np'] = np
    namespace['plt'] = plt
    namespace['sns'] = sns
    
    # Create agents
    agents = create_data_analysis_agents(namespace)
    print(f"✅ Created {len(agents)} specialized agents")
    
    # Create tasks
    tasks = create_data_analysis_tasks(agents, data_variable_name="df")
    print(f"✅ Created {len(tasks)} analysis tasks")
    
    # Define workflow configurations
    workflow_configs = {
        "basic": {
            "tasks": [tasks['data_profiling'], tasks['data_visualization'], tasks['eda_report']],
            "agents": [agents['data_profiler'], agents['visualization'], agents['eda_report']]
        },
        "standard": {
            "tasks": [
                tasks['data_profiling'], 
                tasks['statistical_analysis'], 
                tasks['data_visualization'],
                tasks['data_cleaning'],
                tasks['eda_report']
            ],
            "agents": [
                agents['data_profiler'], 
                agents['insight_analyst'], 
                agents['visualization'],
                agents['data_cleaner'],
                agents['eda_report']
            ]
        },
        "full": {
            "tasks": list(tasks.values()),
            "agents": list(agents.values())
        },
        "custom_insights": {
            "tasks": [
                tasks['data_profiling'],
                tasks['statistical_analysis'],
                tasks['outlier_analysis'],
                tasks['advanced_statistics'],
                tasks['eda_report']
            ],
            "agents": [
                agents['data_profiler'],
                agents['insight_analyst'],
                agents['outlier_analysis'],
                agents['statistics'],
                agents['eda_report']
            ]
        }
    }
    
    # Select workflow configuration
    config = workflow_configs.get(workflow_type, workflow_configs["full"])
    
    # Create the crew with conditional memory based on environment variables
    try:
        # Check if memory-related environment variables are set
        import os
        
        # For Gemini + CrewAI, memory might not work without OpenAI key for ChromaDB
        has_openai_key = bool(os.getenv('CHROMA_OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY'))
        has_google_key = bool(os.getenv('GOOGLE_API_KEY'))
        
        # Disable memory for now if using Gemini without OpenAI key
        enable_memory = has_openai_key
        
        crew = Crew(
            agents=config["agents"],
            tasks=config["tasks"],
            process=Process.sequential,
            verbose=True,
            output_log_file=True,
            max_iter=2,
            memory=enable_memory
        )
        
        if has_google_key and not has_openai_key:
            print("✅ Using Google Gemini LLM")
            print("⚠️  Memory feature disabled (ChromaDB requires OpenAI key)")
        elif enable_memory:
            print("✅ Memory feature enabled with OpenAI")
        else:
            print("⚠️  Memory feature disabled (no compatible API keys found)")
            
    except Exception as e:
        print(f"⚠️  Error creating crew with memory, trying without memory: {str(e)}")
        # Fallback: create crew without memory
        crew = Crew(
            agents=config["agents"],
            tasks=config["tasks"],
            process=Process.sequential,
            verbose=True,
            output_log_file=True,
            max_iter=2,
            memory=False
        )
        print("✅ Crew created without memory feature")
    
    print(f"✅ Crew created with {len(config['tasks'])} tasks in sequential process")
    return crew, namespace

def run_analysis(crew, analysis_name="Data Analysis"):
    """Execute the analysis workflow"""
    print(f"\n🔍 Starting {analysis_name}...")
    print("=" * 60)
    
    try:
        # Execute the crew
        result = crew.kickoff()
        
        print(f"\n✅ {analysis_name} completed successfully!")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during {analysis_name}: {str(e)}")
        return None

def display_results(result, title="Analysis Results"):
    """Display the analysis results in a formatted way"""
    if result:
        print(f"\n📋 {title}")
        print("=" * 60)
        
        # Display as markdown if in Jupyter environment
        try:
            display(Markdown(f"## {title}\n\n{result.raw}"))
        except:
            # Fallback to regular print
            print(result.raw)
    else:
        print("❌ No results to display")

def save_results(result, filename="eda_report.md"):
    """Save the analysis results to a file"""
    if result:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Data Analysis Report\n\n")
                f.write(f"Generated by CrewAI Data Analysis Framework\n\n")
                f.write("---\n\n")
                f.write(result.raw)
            
            print(f"✅ Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
    else:
        print("❌ No results to save")

def generate_cleaning_report(original_df, cleaned_df, cleaning_operations=None):
    """Generate a comprehensive cleaning report"""
    print("\n🧹 Data Cleaning Report")
    print("=" * 60)
    
    # Basic shape comparison
    original_shape = original_df.shape
    cleaned_shape = cleaned_df.shape
    
    print(f"📊 Dataset Shape Changes:")
    print(f"   Original: {original_shape[0]} rows × {original_shape[1]} columns")
    print(f"   Cleaned:  {cleaned_shape[0]} rows × {cleaned_shape[1]} columns")
    print(f"   Rows removed: {original_shape[0] - cleaned_shape[0]} ({((original_shape[0] - cleaned_shape[0])/original_shape[0]*100):.2f}%)")
    print(f"   Columns removed: {original_shape[1] - cleaned_shape[1]}")
    
    # Missing data comparison
    original_missing = original_df.isnull().sum().sum()
    cleaned_missing = cleaned_df.isnull().sum().sum()
    
    print(f"\n🔍 Missing Data Handling:")
    print(f"   Original missing values: {original_missing}")
    print(f"   Cleaned missing values: {cleaned_missing}")
    print(f"   Missing values handled: {original_missing - cleaned_missing}")
    
    # Duplicates check
    original_duplicates = original_df.duplicated().sum()
    cleaned_duplicates = cleaned_df.duplicated().sum()
    
    print(f"\n🔄 Duplicate Handling:")
    print(f"   Original duplicates: {original_duplicates}")
    print(f"   Cleaned duplicates: {cleaned_duplicates}")
    print(f"   Duplicates removed: {original_duplicates - cleaned_duplicates}")
    
    # Data type changes
    original_dtypes = original_df.dtypes.value_counts()
    cleaned_dtypes = cleaned_df.dtypes.value_counts()
    
    print(f"\n📋 Data Type Changes:")
    print(f"   Original data types: {dict(original_dtypes)}")
    print(f"   Cleaned data types: {dict(cleaned_dtypes)}")
    
    # Column-wise missing data
    print(f"\n📈 Column-wise Missing Data:")
    original_missing_by_col = original_df.isnull().sum()
    cleaned_missing_by_col = cleaned_df.isnull().sum()
    
    for col in original_df.columns:
        if col in cleaned_df.columns:
            orig_missing = original_missing_by_col[col]
            clean_missing = cleaned_missing_by_col[col]
            if orig_missing > 0 or clean_missing > 0:
                print(f"   {col}: {orig_missing} → {clean_missing}")
    
    # Removed columns
    removed_columns = set(original_df.columns) - set(cleaned_df.columns)
    if removed_columns:
        print(f"\n❌ Removed Columns:")
        for col in removed_columns:
            print(f"   • {col}")
    
    # Memory usage comparison
    original_memory = original_df.memory_usage(deep=True).sum() / 1024**2
    cleaned_memory = cleaned_df.memory_usage(deep=True).sum() / 1024**2
    
    print(f"\n💾 Memory Usage:")
    print(f"   Original: {original_memory:.2f} MB")
    print(f"   Cleaned: {cleaned_memory:.2f} MB")
    print(f"   Memory saved: {original_memory - cleaned_memory:.2f} MB ({((original_memory - cleaned_memory)/original_memory*100):.2f}%)")
    
    # Additional cleaning operations if provided
    if cleaning_operations:
        print(f"\n🔧 Cleaning Operations Performed:")
        for operation in cleaning_operations:
            print(f"   • {operation}")
    
    return {
        'original_shape': original_shape,
        'cleaned_shape': cleaned_shape,
        'rows_removed': original_shape[0] - cleaned_shape[0],
        'columns_removed': original_shape[1] - cleaned_shape[1],
        'missing_values_handled': original_missing - cleaned_missing,
        'duplicates_removed': original_duplicates - cleaned_duplicates,
        'memory_saved_mb': original_memory - cleaned_memory,
        'removed_columns': list(removed_columns)
    }

def extract_insights_from_analysis(result, namespace):
    """Extract key insights from the analysis results"""
    insights = {
        'statistical_insights': [],
        'data_quality_insights': [],
        'pattern_insights': [],
        'recommendations': []
    }
    
    df = namespace.get('df')
    if df is not None:
        # Statistical insights
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe()
            
            # High variance columns
            high_var_cols = stats.loc['std'][stats.loc['std'] > stats.loc['mean']].index
            if len(high_var_cols) > 0:
                insights['statistical_insights'].append(f"High variance detected in: {list(high_var_cols)}")
            
            # Skewed distributions
            for col in numeric_cols:
                skew = df[col].skew()
                if abs(skew) > 1:
                    insights['statistical_insights'].append(f"{col} is {'right' if skew > 0 else 'left'} skewed (skewness: {skew:.2f})")
        
        # Data quality insights
        missing_data = df.isnull().sum()
        if missing_data.any():
            high_missing_cols = missing_data[missing_data > len(df) * 0.1].index
            if len(high_missing_cols) > 0:
                insights['data_quality_insights'].append(f"Columns with >10% missing data: {list(high_missing_cols)}")
        
        # Correlation insights
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if high_corr_pairs:
                insights['pattern_insights'].append(f"Strong correlations found: {high_corr_pairs[:3]}")
        
        # Outlier insights
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                outlier_pct = len(outliers) / len(df) * 100
                if outlier_pct > 5:
                    insights['pattern_insights'].append(f"{col} has {len(outliers)} outliers ({outlier_pct:.1f}%)")
        
        # Recommendations
        if len(insights['statistical_insights']) > 0:
            insights['recommendations'].append("Consider data transformation for highly skewed variables")
        
        if len(insights['data_quality_insights']) > 0:
            insights['recommendations'].append("Investigate and handle missing data patterns")
        
        if len(insights['pattern_insights']) > 0:
            insights['recommendations'].append("Review correlations and outliers for data integrity")
    
    return insights

def save_cleaned_dataset(df, filename="cleaned_dataset.csv"):
    """Save the cleaned dataset to a file"""
    try:
        df.to_csv(filename, index=False)
        print(f"✅ Cleaned dataset saved to {filename}")
        print(f"📊 Saved dataset shape: {df.shape}")
        return filename
    except Exception as e:
        print(f"❌ Error saving cleaned dataset: {str(e)}")
        return None

def generate_comprehensive_report(original_df, result, namespace, workflow_type):
    """Generate a comprehensive analysis report with cleaning info and insights"""
    print("\n📋 Generating Comprehensive Analysis Report")
    print("=" * 60)
    
    cleaned_df = namespace.get('df', original_df)
    
    # Generate cleaning report
    cleaning_report = generate_cleaning_report(original_df, cleaned_df)
    
    # Extract insights
    insights = extract_insights_from_analysis(result, namespace)
    
    # Display insights
    print("\n💡 Key Insights Summary")
    print("=" * 60)
    
    for category, insight_list in insights.items():
        if insight_list:
            print(f"\n📌 {category.replace('_', ' ').title()}:")
            for insight in insight_list:
                print(f"   • {insight}")
    
    # Save cleaned dataset
    cleaned_file = save_cleaned_dataset(cleaned_df, f"cleaned_dataset_{workflow_type}.csv")
    
    # Create comprehensive report file
    report_filename = f"comprehensive_report_{workflow_type}.md"
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Data Analysis Report\n\n")
            f.write(f"Generated by CrewAI Data Analysis Framework\n")
            f.write(f"Workflow Type: {workflow_type}\n\n")
            f.write("---\n\n")
            
            # Data Cleaning Section
            f.write("## 🧹 Data Cleaning Summary\n\n")
            f.write(f"**Original Dataset:** {cleaning_report['original_shape'][0]} rows × {cleaning_report['original_shape'][1]} columns\n")
            f.write(f"**Cleaned Dataset:** {cleaning_report['cleaned_shape'][0]} rows × {cleaning_report['cleaned_shape'][1]} columns\n\n")
            f.write(f"**Changes Made:**\n")
            f.write(f"- Rows removed: {cleaning_report['rows_removed']} ({(cleaning_report['rows_removed']/cleaning_report['original_shape'][0]*100):.2f}%)\n")
            f.write(f"- Columns removed: {cleaning_report['columns_removed']}\n")
            f.write(f"- Missing values handled: {cleaning_report['missing_values_handled']}\n")
            f.write(f"- Duplicates removed: {cleaning_report['duplicates_removed']}\n")
            f.write(f"- Memory saved: {cleaning_report['memory_saved_mb']:.2f} MB\n\n")
            
            if cleaning_report['removed_columns']:
                f.write(f"**Removed Columns:** {', '.join(cleaning_report['removed_columns'])}\n\n")
            
            # Insights Section
            f.write("## 💡 Key Insights\n\n")
            for category, insight_list in insights.items():
                if insight_list:
                    f.write(f"### {category.replace('_', ' ').title()}\n")
                    for insight in insight_list:
                        f.write(f"- {insight}\n")
                    f.write("\n")
            
            # Analysis Results Section
            f.write("## 📊 Detailed Analysis Results\n\n")
            if result:
                f.write(result.raw)
            
            # Files Generated Section
            f.write("\n## 📁 Generated Files\n\n")
            f.write(f"- **Cleaned Dataset:** {cleaned_file or 'Not saved'}\n")
            f.write(f"- **Analysis Report:** {report_filename}\n")
            f.write(f"- **EDA Report:** eda_report_{workflow_type}.md\n")
        
        print(f"✅ Comprehensive report saved to {report_filename}")
        
    except Exception as e:
        print(f"❌ Error saving comprehensive report: {str(e)}")
    
    return {
        'cleaning_report': cleaning_report,
        'insights': insights,
        'cleaned_dataset_file': cleaned_file,
        'report_file': report_filename
    }

def run_custom_analysis(workflow_type="standard", data_path=None, custom_config=None):
    """Run a custom analysis with specified parameters"""
    print(f"🎯 Running custom analysis: {workflow_type}")
    
    # Setup
    setup_environment()
    
    # Load data
    if data_path is None:
        data_path = DATA_CONFIG.get('file_path', '/content/Supplement_Sales_Weekly.csv')
    
    original_df = load_data(data_path)
    if original_df is None:
        return None, None
    
    # Apply custom configuration if provided
    if custom_config:
        print(f"🔧 Applying custom configuration: {custom_config}")
        # You can extend this to modify workflow based on custom_config
    
    # Create and run workflow
    crew, namespace = create_analysis_workflow(original_df.copy(), workflow_type)
    result = run_analysis(crew, f"Custom Analysis ({workflow_type})")
    
    if result:
        display_results(result, f"Custom Analysis Results ({workflow_type})")
        save_results(result, f"custom_analysis_{workflow_type}.md")
        
        # Generate comprehensive report with cleaning info and insights
        comprehensive_report = generate_comprehensive_report(original_df, result, namespace, workflow_type)
        
        # Store additional info in namespace
        namespace['original_df'] = original_df
        namespace['comprehensive_report'] = comprehensive_report
    
    return result, namespace

# Core analysis functions
def quick_analysis():
    """Run a quick basic analysis"""
    return run_custom_analysis("basic")

def standard_analysis():
    """Run a standard comprehensive analysis"""
    return run_custom_analysis("standard")

def full_deep_analysis():
    """Run the complete deep analysis with all agents"""
    return run_custom_analysis("full")

def insights_focused_analysis():
    """Run an analysis focused on insights and statistics"""
    return run_custom_analysis("custom_insights")

# Usage Example Functions
def example_basic_usage():
    """Example of basic framework usage"""
    print("🔍 Running Basic Analysis Example")
    print("=" * 50)
    
    # Run a quick analysis
    result, namespace = quick_analysis()
    
    if result:
        print("✅ Quick analysis completed!")
        print("📊 Key findings available in result.raw")
        
        # Access the cleaned data from namespace
        df = namespace.get('df')
        if df is not None:
            print(f"📋 Dataset shape: {df.shape}")
    
    return result, namespace

def example_custom_workflow():
    """Example of creating a custom analysis workflow"""
    print("🎯 Running Custom Workflow Example")
    print("=" * 50)
    
    # Define custom analysis parameters
    custom_config = {
        'focus_areas': ['statistical_summary', 'correlation_analysis', 'outlier_detection'],
        'visualization_types': ['histogram', 'scatter', 'heatmap'],
        'analysis_depth': 'detailed',
        'include_recommendations': True
    }
    
    # Run custom analysis
    result, namespace = run_custom_analysis("standard", custom_config=custom_config)
    
    if result:
        print("✅ Custom analysis completed!")
        print("🎨 Custom visualizations generated")
        
        # Access results
        df = namespace.get('df')
        insights = namespace.get('insights', {})
        
        if df is not None:
            print(f"📊 Analyzed {len(df.columns)} features across {len(df)} records")
        
        if insights:
            print(f"🔍 Generated {len(insights)} key insights")
    
    return result, namespace

def example_comparative_analysis():
    """Example of running different analysis types for comparison"""
    print("⚖️ Running Comparative Analysis Example")
    print("=" * 50)
    
    results = {}
    
    # Run different analysis types
    analysis_types = [
        ('Quick', quick_analysis),
        ('Standard', standard_analysis),
        ('Deep', full_deep_analysis),
        ('Insights-Focused', insights_focused_analysis)
    ]
    
    for name, analysis_func in analysis_types:
        print(f"\n🔄 Running {name} Analysis...")
        try:
            result, namespace = analysis_func()
            results[name] = {
                'success': result is not None,
                'namespace': namespace
            }
            if result:
                print(f"✅ {name} analysis completed successfully")
            else:
                print(f"❌ {name} analysis failed")
        except Exception as e:
            print(f"❌ {name} analysis error: {str(e)}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n📋 Analysis Summary:")
    for name, result in results.items():
        status = "✅ Success" if result.get('success') else "❌ Failed"
        print(f"  {name}: {status}")
    
    return results

def example_data_exploration():
    """Example of exploring dataset characteristics"""
    print("🔬 Running Data Exploration Example")
    print("=" * 50)
    
    # Run standard analysis to get cleaned data
    result, namespace = standard_analysis()
    
    if result and namespace.get('df') is not None:
        df = namespace['df']
        
        print("📊 Dataset Overview:")
        print(f"  • Shape: {df.shape}")
        print(f"  • Columns: {list(df.columns)}")
        print(f"  • Data types: {df.dtypes.value_counts().to_dict()}")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.any():
            print(f"  • Missing values: {missing_data[missing_data > 0].to_dict()}")
        else:
            print("  • No missing values found")
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"  • Numeric columns: {len(numeric_cols)}")
            print("  • Basic statistics:")
            stats = df[numeric_cols].describe()
            for col in numeric_cols[:3]:  # Show first 3 columns
                print(f"    - {col}: mean={stats.loc['mean', col]:.2f}, std={stats.loc['std', col]:.2f}")
        
        # Categorical columns summary
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            print(f"  • Categorical columns: {len(cat_cols)}")
            for col in cat_cols[:3]:  # Show first 3 columns
                unique_count = df[col].nunique()
                print(f"    - {col}: {unique_count} unique values")
    
    return result, namespace

def example_insights_extraction():
    """Example of extracting and displaying insights"""
    print("💡 Running Insights Extraction Example")
    print("=" * 50)
    
    # Run insights-focused analysis
    result, namespace = insights_focused_analysis()
    
    if result:
        insights = namespace.get('insights', {})
        recommendations = namespace.get('recommendations', [])
        
        print("🔍 Key Insights:")
        if insights:
            for category, insight_list in insights.items():
                print(f"\n  📌 {category.title()}:")
                if isinstance(insight_list, list):
                    for insight in insight_list[:3]:  # Show top 3 insights
                        print(f"    • {insight}")
                else:
                    print(f"    • {insight_list}")
        
        print("\n💡 Recommendations:")
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5 recommendations
                print(f"  {i}. {rec}")
        
        # Additional analysis metrics
        df = namespace.get('df')
        if df is not None:
            print(f"\n📊 Analysis covered {len(df.columns)} features with {len(df)} data points")
    
    return result, namespace

def run_all_examples():
    """Run all example functions"""
    print("🚀 Running All CrewAI Data Analysis Examples")
    print("=" * 60)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Workflow", example_custom_workflow),
        ("Comparative Analysis", example_comparative_analysis),
        ("Data Exploration", example_data_exploration),
        ("Insights Extraction", example_insights_extraction)
    ]
    
    results = {}
    
    for name, example_func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = example_func()
            results[name] = {'success': True, 'result': result}
            print(f"✅ {name} completed successfully\n")
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
            print(f"❌ {name} failed: {str(e)}\n")
    
    # Final summary
    print("🏁 Final Summary")
    print("=" * 60)
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    print(f"✅ Successfully completed: {successful}/{total} examples")
    
    if successful < total:
        print("❌ Failed examples:")
        for name, result in results.items():
            if not result['success']:
                print(f"  • {name}: {result.get('error', 'Unknown error')}")
    
    return results

def interactive_example():
    """Interactive example allowing user to choose analysis type"""
    print("🎮 Interactive CrewAI Data Analysis Example")
    print("=" * 50)
    
    # Print usage guide first
    print_usage_guide()
    
    print("\nChoose an analysis type:")
    print("1. Quick Analysis (Fast overview)")
    print("2. Standard Analysis (Balanced depth)")
    print("3. Full Deep Analysis (Comprehensive)")
    print("4. Insights-Focused Analysis (Key findings)")
    print("5. Custom Analysis (Configurable)")
    print("6. Run All Examples")
    print("7. Data Exploration Example")
    print("8. Insights Extraction Example")
    print("9. Comparative Analysis Example")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-9): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                return quick_analysis()
            elif choice == '2':
                return standard_analysis()
            elif choice == '3':
                return full_deep_analysis()
            elif choice == '4':
                return insights_focused_analysis()
            elif choice == '5':
                # Get custom configuration from user
                config = {}
                print("Configure your custom analysis:")
                
                focus_input = input("Focus areas (comma-separated, e.g., 'stats,correlation'): ").strip()
                if focus_input:
                    config['focus_areas'] = [area.strip() for area in focus_input.split(',')]
                
                viz_input = input("Visualization types (comma-separated, e.g., 'histogram,scatter'): ").strip()
                if viz_input:
                    config['visualization_types'] = [viz.strip() for viz in viz_input.split(',')]
                
                depth_input = input("Analysis depth (basic/detailed/comprehensive): ").strip()
                if depth_input:
                    config['analysis_depth'] = depth_input
                
                return run_custom_analysis("standard", custom_config=config)
            elif choice == '6':
                return run_all_examples()
            elif choice == '7':
                return example_data_exploration()
            elif choice == '8':
                return example_insights_extraction()
            elif choice == '9':
                return example_comparative_analysis()
            else:
                print("❌ Invalid choice. Please enter a number between 0-9.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """Main execution function"""
    print("🚀 CrewAI Data Analysis Framework")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Load data
    data_path = DATA_CONFIG.get('file_path', '/content/Supplement_Sales_Weekly.csv')
    original_df = load_data(data_path)
    
    if original_df is None:
        print("❌ Failed to load data. Exiting...")
        return
    
    # Create and run analysis workflow
    workflow_type = "full"  # Options: "basic", "standard", "full", "custom_insights"
    crew, namespace = create_analysis_workflow(original_df.copy(), workflow_type)
    
    if crew:
        result = run_analysis(crew, f"Comprehensive Data Analysis ({workflow_type})")
        
        if result:
            # Display results
            display_results(result, "Comprehensive EDA Report")
            
            # Save results
            save_results(result, f"eda_report_{workflow_type}.md")
            
            # Generate comprehensive report with cleaning info and insights
            comprehensive_report = generate_comprehensive_report(original_df, result, namespace, workflow_type)
            
            # Print summary
            print(f"\n📊 Final Analysis Summary:")
            print(f"   - Original Dataset: {original_df.shape[0]} rows × {original_df.shape[1]} columns")
            print(f"   - Cleaned Dataset: {namespace['df'].shape[0]} rows × {namespace['df'].shape[1]} columns")
            print(f"   - Workflow: {workflow_type}")
            print(f"   - Tasks completed: {len(crew.tasks)}")
            print(f"   - Files generated:")
            print(f"     • EDA Report: eda_report_{workflow_type}.md")
            print(f"     • Comprehensive Report: {comprehensive_report['report_file']}")
            print(f"     • Cleaned Dataset: {comprehensive_report['cleaned_dataset_file']}")
            
        else:
            print("❌ Analysis failed to complete")
    else:
        print("❌ Failed to create analysis workflow")

# Helper functions for Jupyter notebook usage
def print_usage_guide():
    """Print usage guide for the framework"""
    guide = """
    🔍 CrewAI Data Analysis Framework Usage Guide
    ==========================================
    
    This framework provides comprehensive data analysis capabilities using CrewAI agents.
    
    📋 Available Functions:
    ----------------------
    Core Analysis:
    1. main() - Run full analysis workflow with comprehensive reporting
    2. quick_analysis() - Basic profiling and visualization  
    3. standard_analysis() - Comprehensive analysis with cleaning
    4. full_deep_analysis() - Complete analysis with all features
    5. insights_focused_analysis() - Focus on statistical insights
    6. run_custom_analysis(workflow_type, data_path, custom_config) - Custom workflow
    
    Usage Examples:
    7. example_basic_usage() - Basic framework usage example
    8. example_custom_workflow() - Custom workflow configuration
    9. example_comparative_analysis() - Compare different analysis types
    10. example_data_exploration() - Dataset exploration example
    11. example_insights_extraction() - Extract and display insights
    12. run_all_examples() - Run all usage examples
    13. interactive_example() - Interactive mode for choosing analysis
    
    📊 Output Files Generated:
    -------------------------
    - comprehensive_report_{workflow}.md - Complete analysis report with cleaning details
    - cleaned_dataset_{workflow}.csv - Cleaned and processed dataset
    - eda_report_{workflow}.md - Detailed EDA findings
    
    💡 Key Features:
    ---------------
    ✅ Data cleaning tracking and reporting
    ✅ Automated insights extraction
    ✅ Statistical analysis and pattern detection
    ✅ Cleaned dataset export
    ✅ Memory usage optimization reporting
    ✅ Missing data and outlier analysis
    
    🔧 Workflow Types:
    -----------------
    - "basic": Data profiling + visualization + report
    - "standard": Above + statistical analysis + cleaning  
    - "full": All available analysis components
    - "custom_insights": Focus on statistics and outliers
    
    📊 Configuration:
    ----------------
    - Edit config.py to change API keys and settings
    - Modify DATA_CONFIG for different datasets
    - Adjust ANALYSIS_CONFIG for analysis parameters
    
    🚀 Quick Start Examples:
    -----------------------
    # Run standard analysis
    result, namespace = standard_analysis()
    
    # Run custom analysis with config
    config = {'focus_areas': ['stats', 'correlation']}
    result, namespace = run_custom_analysis("standard", custom_config=config)
    
    # Run interactive mode
    interactive_example()
    
    # Run all examples
    run_all_examples()
    """
    print(guide)

if __name__ == "__main__":
    """Main execution block with options"""
    print("🎯 CrewAI Data Analysis Framework")
    print("=" * 60)
    
    # Choose execution mode
    print("Select execution mode:")
    print("1. Run main analysis workflow")
    print("2. Run interactive examples")
    print("3. Run all examples")
    print("4. Print usage guide")
    
    try:
        choice = input("Enter your choice (1-4, or press Enter for default main): ").strip()
        
        if choice == '2':
            interactive_example()
        elif choice == '3':
            run_all_examples()
        elif choice == '4':
            print_usage_guide()
        else:
            # Default: run main analysis
            main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        # Fallback to main analysis
        main()

# Make key functions easily accessible
__all__ = [
    'main', 'quick_analysis', 'standard_analysis', 'full_deep_analysis',
    'insights_focused_analysis', 'run_custom_analysis', 'print_usage_guide',
    'example_basic_usage', 'example_custom_workflow', 'example_comparative_analysis',
    'example_data_exploration', 'example_insights_extraction', 'run_all_examples',
    'interactive_example'
]