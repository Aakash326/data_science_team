# 🚀 Advanced AI Data Analysis Platform - Setup Guide

## 🎯 Overview
This is a comprehensive AI-powered data analysis platform that combines:
- **CrewAI** framework with 8 specialized AI agents
- **OpenAI GPT models** for intelligent insights
- **Streamlit** web interface for easy use
- **Advanced data processing** with smart missing value handling

## 📋 Prerequisites

### 1. System Requirements
- Python 3.8 or higher
- Internet connection for AI model access
- Modern web browser for Streamlit interface

### 2. OpenAI API Key
You need a valid OpenAI API key with sufficient credits:
1. Visit [OpenAI Platform](https://platform.openai.com/account/api-keys)
2. Create a new API key
3. Copy the key (starts with `sk-proj-...`)

## ⚙️ Installation & Setup

### Step 1: Install Dependencies
```bash
cd data_analysis_team/src
pip install -r requirements.txt
```

### Step 2: Configure OpenAI API Key
Create a `.env` file in the `src` directory:
```bash
echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
```

**Replace `your_actual_api_key_here` with your actual OpenAI API key.**

### Step 3: Verify Setup
Run the demo script to test everything:
```bash
python demo_usage.py
```

If successful, you should see:
- ✅ Sample dataset created
- ✅ OpenAI API key validated
- 🚀 Analysis workflow demonstration

## 🌐 Launch the Web Application

### Option 1: Direct Launch
```bash
streamlit run app.py
```

### Option 2: Using Launch Script
```bash
chmod +x launch_app.sh
./launch_app.sh
```

The application will open at: **http://localhost:8501**

## 🔧 Usage Guide

### 1. Data Upload & Configuration
- Go to "📤 Data Upload & Configuration" tab
- Upload your CSV file (drag & drop or browse)
- Review dataset preview and statistics
- Configure analysis parameters:
  - **Target column**: Main column to analyze/predict
  - **Date column**: For time-series analysis (optional)
  - **Column types**: Classify as categorical or numerical
  - **Analysis type**: Quick, Standard, or Advanced
  - **Thresholds**: Missing values and correlation limits

### 2. Run Analysis
- Click "🚀 Configure & Run Analysis"
- Watch real-time progress as 8 AI agents work:
  - Data Quality Specialist
  - Statistical Intelligence Analyst  
  - Correlation & Feature Engineer
  - Business Intelligence Strategist
  - Visualization Storytelling Expert
  - Predictive Modeling Consultant
  - Advanced Analytics Researcher
  - Master Data Science Orchestrator

### 3. View Results
- Go to "🔬 Analysis Results" tab
- Review comprehensive insights and recommendations
- Explore statistical analysis and correlations
- Check data quality improvements

### 4. Download Outputs
- **Cleaned Dataset**: CSV file with processed data
- **Analysis Report**: Comprehensive markdown report
- **Configuration**: Save settings for future use

## 🎯 Analysis Types

### Quick Analysis (1-2 minutes)
- Basic data profiling
- Missing value treatment
- Simple statistical insights
- Ideal for initial exploration

### Standard Analysis (3-5 minutes)
- Comprehensive data quality assessment
- Advanced statistical analysis
- Correlation analysis
- Business insights and recommendations

### Advanced Analysis (5-10 minutes)
- Deep statistical investigation
- Predictive modeling insights
- Advanced feature engineering
- Comprehensive business intelligence
- Detailed visualization recommendations

## 📊 Supported Data Types

### File Formats
- **CSV files** (primary support)
- UTF-8, ASCII, Latin-1 encoding detection
- Automatic delimiter detection

### Column Types
- **Numerical**: integers, floats, decimals
- **Categorical**: strings, categories, labels
- **DateTime**: dates, timestamps, time series
- **Boolean**: true/false, yes/no values

### Data Sizes
- **Small datasets**: < 1MB (instant processing)
- **Medium datasets**: 1-10MB (recommended)
- **Large datasets**: 10-100MB (use Quick analysis first)
- **Very large datasets**: > 100MB (consider sampling)

## 🛠️ Troubleshooting

### OpenAI API Issues
```
❌ OpenAI API test failed: Incorrect API key provided
```
**Solution**: Check your `.env` file and ensure the API key is correct.

### Memory Issues
```
❌ Memory error or slow processing
```
**Solution**: 
- Use "Quick" analysis for large datasets
- Consider data sampling for datasets > 100MB
- Increase system memory if possible

### Import Errors
```
❌ ModuleNotFoundError: No module named 'X'
```
**Solution**: Install missing packages:
```bash
pip install -r requirements.txt
```

### Port Already in Use
```
❌ Port 8501 is already in use
```
**Solution**: Use a different port:
```bash
streamlit run app.py --server.port 8502
```

## 🔒 Security & Privacy

### Data Handling
- **Local Processing**: Your data stays on your machine
- **API Calls**: Only metadata and statistical summaries sent to OpenAI
- **No Raw Data**: Original data never leaves your environment

### API Security
- Store API keys in `.env` file (never commit to version control)
- API key is validated but never logged in full
- Use environment variables for production deployment

## 🚀 Advanced Features

### Programmatic Usage
```python
from main import AdvancedDataAnalyzer

analyzer = AdvancedDataAnalyzer()
results = analyzer.analyze_data(
    'your_data.csv',
    config={
        'target_column': 'target',
        'analysis_type': 'advanced'
    }
)
```

### Custom Configuration
```python
config = {
    'target_column': 'sales',
    'date_column': 'date',
    'categorical_columns': ['category', 'region'],
    'numerical_columns': ['price', 'quantity'],
    'missing_value_threshold': 0.05,
    'correlation_threshold': 0.7,
    'workflow_type': 'advanced'
}
```

### Batch Processing
```python
# Process multiple files
files = ['data1.csv', 'data2.csv', 'data3.csv']
for file in files:
    results = analyzer.analyze_data(file, config)
    # Save results
```

## 📈 Performance Tips

### For Large Datasets
1. Start with "Quick" analysis
2. Use column sampling for initial exploration
3. Focus on specific columns of interest
4. Consider data preprocessing before upload

### For Best Results
1. Clean column names (no special characters)
2. Ensure consistent data formats
3. Handle obvious data quality issues first
4. Provide clear target column for analysis

### Cost Optimization
1. Use "Quick" analysis for exploration
2. Reserve "Advanced" for final analysis
3. Process smaller, focused datasets
4. Monitor OpenAI API usage

## 📞 Support

### Common Issues
- Check the logs in the terminal for detailed error messages
- Verify internet connection for AI model access
- Ensure sufficient OpenAI API credits
- Try restarting the Streamlit app

### Getting Help
1. Review error messages in terminal
2. Check this setup guide
3. Verify all prerequisites are met
4. Test with the demo script first

---

🎉 **You're ready to unlock the power of AI-driven data analysis!**

Launch the app with `streamlit run app.py` and start exploring your data with intelligent insights.
