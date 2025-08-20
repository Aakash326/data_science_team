# basic app.py
import streamlit as st
import tempfile
import os
import pandas as pd
from main import AdvancedDataAnalyzer  # Import your analyzer from main.py
import json

st.set_page_config(page_title="Advanced Data Analyzer", layout="wide")
st.title("🔬 Advanced Data Analysis with OpenAI")

# File uploader
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    # Save uploaded file to temp path
    suffix = ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Read uploaded file
    try:
        df = pd.read_csv(file_path)
        st.success(f"✅ Dataset loaded successfully with shape: {df.shape}")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.stop()

    # Column list
    columns = list(df.columns)

    st.subheader("⚙️ Configure DATA_CONFIG")
    target_column = st.text_input("Target Column", value="")
    date_column = st.text_input("Date Column (leave blank if none)", value="")
    categorical_columns = st.multiselect(
        "Categorical Columns", options=columns, default=[]
    )
    numerical_columns = st.multiselect(
        "Numerical Columns", options=columns, default=[]
    )

    # Assemble DATA_CONFIG
    DATA_CONFIG = {
        "file_path": file_path,
        "target_column": target_column,
        "date_column": date_column,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns
    }

    # Show config
    st.write("**Current DATA_CONFIG:**")
    st.json(DATA_CONFIG)

    # Save config to file
    if st.button("💾 Save DATA_CONFIG"):
        with open("config.py", "w") as f:
            f.write(f"DATA_CONFIG = {json.dumps(DATA_CONFIG, indent=4)}\n")
        st.success("✅ DATA_CONFIG saved to config.py")

    # Validate and run analysis
    if st.button("🚀 Run Analysis"):
        if not target_column:
            st.error("❌ Please specify a target column before running analysis.")
        elif target_column not in df.columns:
            st.error("❌ Target column not found in dataset.")
        else:
            with st.spinner("Running Advanced Data Analyzer..."):
                analyzer = AdvancedDataAnalyzer()
                result = analyzer.run_analysis(file_path, workflow_type="advanced")

                if result.get("success"):
                    st.success("✅ Analysis completed successfully!")

                    # Cleaned dataset
                    cleaned_df = result.get("cleaned_data")
                    cleaned_csv = cleaned_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Cleaned Dataset",
                        data=cleaned_csv,
                        file_name="cleaned_dataset.csv",
                        mime="text/csv"
                    )

                    # EDA report file
                    report_info = result.get("report", {})
                    report_file = report_info.get("report_file")
                    if report_file and os.path.exists(report_file):
                        with open(report_file, "rb") as f:
                            st.download_button(
                                label="⬇️ Download EDA Report",
                                data=f,
                                file_name=report_file,
                                mime="text/markdown"
                            )

                    # Display missing value log
                    st.subheader("🔧 Missing Value Treatment Log")
                    cleaning_log = result.get("cleaning_log", [])
                    if cleaning_log:
                        st.dataframe(pd.DataFrame(cleaning_log))
                    else:
                        st.info("No missing values were treated.")

                else:
                    st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")