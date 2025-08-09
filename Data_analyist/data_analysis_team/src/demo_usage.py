# demo_usage.py - Demonstration of the Streamlit App Features

"""
This script demonstrates how to use the Advanced AI Data Analysis Platform
both through the Streamlit interface and programmatically.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_dataset():
    """Create a sample dataset for demonstration purposes."""
    np.random.seed(42)
    
    # Generate sample e-commerce data
    n_rows = 1000
    
    data = {
        'customer_id': np.random.randint(1000, 9999, n_rows),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_rows),
        'purchase_amount': np.random.normal(100, 50, n_rows),
        'quantity': np.random.randint(1, 10, n_rows),
        'discount_percentage': np.random.uniform(0, 0.3, n_rows),
        'customer_age': np.random.randint(18, 80, n_rows),
        'customer_location': np.random.choice(['New York', 'California', 'Texas', 'Florida', 'Illinois'], n_rows),
        'purchase_date': [
            datetime.now() - timedelta(days=int(x)) 
            for x in np.random.randint(0, 365, n_rows)
        ],
        'satisfaction_score': np.random.uniform(1, 5, n_rows),
        'is_premium_customer': np.random.choice([True, False], n_rows, p=[0.3, 0.7])
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values to demonstrate cleaning capabilities
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices[:len(missing_indices)//3], 'customer_age'] = np.nan
    df.loc[missing_indices[len(missing_indices)//3:2*len(missing_indices)//3], 'satisfaction_score'] = np.nan
    df.loc[missing_indices[2*len(missing_indices)//3:], 'discount_percentage'] = np.nan
    
    # Ensure purchase amounts are positive
    df['purchase_amount'] = np.abs(df['purchase_amount'])
    
    return df

def save_sample_dataset():
    """Create and save a sample dataset for demo purposes."""
    df = create_sample_dataset()
    filename = "sample_ecommerce_data.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Sample dataset created: {filename}")
    print(f"📊 Dataset shape: {df.shape}")
    print("\n📋 Dataset Preview:")
    print(df.head())
    print("\n📈 Dataset Info:")
    print(df.info())
    return filename

def demonstrate_config_options():
    """Demonstrate different configuration options for the analysis."""
    
    # Configuration for e-commerce analysis
    ecommerce_config = {
        "target_column": "purchase_amount",
        "date_column": "purchase_date",
        "categorical_columns": [
            "product_category", 
            "customer_location", 
            "is_premium_customer"
        ],
        "numerical_columns": [
            "customer_id",
            "quantity", 
            "discount_percentage", 
            "customer_age", 
            "satisfaction_score"
        ],
        "missing_value_threshold": 0.05,
        "correlation_threshold": 0.7,
        "workflow_type": "advanced"
    }
    
    # Configuration for quick analysis
    quick_config = {
        "target_column": "satisfaction_score",
        "date_column": "",
        "categorical_columns": ["product_category", "customer_location"],
        "numerical_columns": ["purchase_amount", "customer_age"],
        "missing_value_threshold": 0.10,
        "correlation_threshold": 0.8,
        "workflow_type": "quick"
    }
    
    print("🔧 Sample Configuration Options:")
    print("\n1. E-commerce Revenue Analysis:")
    for key, value in ecommerce_config.items():
        print(f"   {key}: {value}")
    
    print("\n2. Customer Satisfaction Analysis:")
    for key, value in quick_config.items():
        print(f"   {key}: {value}")
    
    return ecommerce_config, quick_config

def print_streamlit_usage_guide():
    """Print a step-by-step guide for using the Streamlit app."""
    
    guide = """
🚀 Streamlit App Usage Guide
===========================

1. 📁 PREPARE YOUR DATA
   ────────────────────
   • Ensure your data is in CSV format
   • Check that column names are clear and descriptive
   • Verify data quality (minimal missing values preferred)

2. 🌐 LAUNCH THE APP
   ─────────────────
   • Run: streamlit run app.py
   • Or use: ./launch_app.sh
   • Open browser to http://localhost:8501

3. 📤 UPLOAD & CONFIGURE
   ────────────────────
   • Go to "📤 Data Upload & Configuration" tab
   • Drag and drop your CSV file
   • Review the dataset preview and statistics

4. ⚙️ SET ANALYSIS PARAMETERS
   ─────────────────────────
   • Select target column (what you want to predict/analyze)
   • Choose date column (for time-series analysis, optional)
   • Classify columns as categorical or numerical
   • Adjust analysis type: Quick, Standard, or Advanced
   • Set thresholds for missing values and correlations

5. 🚀 RUN ANALYSIS
   ──────────────
   • Click "🚀 Configure & Run Analysis"
   • Watch real-time progress as AI agents work
   • Wait for completion (may take 2-5 minutes for advanced analysis)

6. 📊 VIEW RESULTS
   ──────────────
   • Go to "🔬 Analysis Results" tab
   • Review data quality improvements
   • Explore AI-generated insights and recommendations
   • Check statistical analysis and correlations

7. 📥 DOWNLOAD OUTPUTS
   ──────────────────
   • Download cleaned dataset (CSV)
   • Download comprehensive analysis report (Markdown)
   • Save configuration file for future use

🎯 PRO TIPS:
──────────
• Start with "Quick" analysis for large datasets
• Use "Advanced" for comprehensive business insights
• Export configurations to reuse on similar datasets
• Check the Settings tab for API and system information

🔧 TROUBLESHOOTING:
─────────────────
• Ensure OpenAI API key is properly configured
• Check internet connection for AI model access
• Verify CSV file format and encoding
• Use smaller datasets for initial testing
"""
    
    print(guide)

def demonstrate_programmatic_usage():
    """Show how to use the analyzer programmatically outside of Streamlit."""
    
    try:
        from main import AdvancedDataAnalyzer
        
        print("\n🤖 Programmatic Usage Example:")
        print("━" * 50)
        
        # Create sample data
        filename = save_sample_dataset()
        
        # Initialize analyzer
        print("\n1. Initializing Advanced Data Analyzer...")
        analyzer = AdvancedDataAnalyzer()
        
        # Run analysis
        print("2. Running comprehensive analysis...")
        result = analyzer.run_analysis(filename, workflow_type="quick")
        
        if result['success']:
            print("✅ Analysis completed successfully!")
            print(f"📊 Original shape: {result['report']['cleaning_stats']['original_shape']}")
            print(f"🧹 Missing values cleaned: {result['report']['cleaning_stats']['missing_before']}")
            print(f"📋 Report file: {result['report']['report_file']}")
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            
    except ImportError:
        print("⚠️ Main analyzer not available. Use the Streamlit app instead.")

def main():
    """Main demonstration function."""
    print("🎯 Advanced AI Data Analysis Platform - Demo")
    print("=" * 60)
    
    # Create sample dataset
    print("\n📊 Creating Sample Dataset...")
    save_sample_dataset()
    
    # Show configuration options
    print("\n⚙️ Configuration Options...")
    demonstrate_config_options()
    
    # Print usage guide
    print_streamlit_usage_guide()
    
    # Demonstrate programmatic usage
    demonstrate_programmatic_usage()
    
    print("\n🚀 Ready to start!")
    print("Launch the Streamlit app with: streamlit run app.py")
    print("Or use the launch script: ./launch_app.sh")

if __name__ == "__main__":
    main()
