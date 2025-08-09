#!/bin/bash
# launch_app.sh - Script to launch the Streamlit app

echo "🚀 Launching Advanced AI Data Analysis Platform..."
echo "=" * 60

# Change to the correct directory
cd /Users/saiaakash/Desktop/All_proj/Data_analyist/data_analysis_team/src

# Check if required files exist
if [[ ! -f "app.py" ]]; then
    echo "❌ Error: app.py not found"
    exit 1
fi

if [[ ! -f "main.py" ]]; then
    echo "❌ Error: main.py not found"
    exit 1
fi

if [[ ! -f "config.py" ]]; then
    echo "❌ Error: config.py not found"
    exit 1
fi

echo "✅ All required files found"
echo "🌐 Starting Streamlit server..."
echo "📱 The app will open in your default browser"
echo "🔧 Use Ctrl+C to stop the server"
echo ""

# Launch Streamlit app
streamlit run app.py --server.port 8501 --server.address localhost
