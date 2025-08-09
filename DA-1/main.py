import logging
from crewai import Crew, Process
from tasks import (
    data_profiling_task,
    insight_extraction_task,
    visualization_task,
    data_cleaning_task,
    eda_report_task,
    statistics_task,
    outlier_analysis_task,
)
from agents import (
    data_profiler_agent,
    insight_analyst_agent,
    visualization_agent,
    data_cleaner_agent,
    eda_report_agent,
    statistics_agent,
    outlier_analysis_agent,
)
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def main():
    try:
        # Specify the path to your CSV file
        csv_file_path = "data.csv"  # Make sure the file name matches your CSV file

        # Load the data
        data = load_data(csv_file_path)

        # Display the first few rows of the dataframe to verify it loaded correctly
        print(data.head())

        # Create and run the crew
        crew = Crew(
            agents=[
                data_profiler_agent,
                insight_analyst_agent,
                visualization_agent,
                data_cleaner_agent,
                eda_report_agent,
                statistics_agent,
                outlier_analysis_agent,
            ],
            tasks=[
                data_profiling_task,
                insight_extraction_task,
                visualization_task,
                data_cleaning_task,
                eda_report_task,
                statistics_task,
                outlier_analysis_task,
            ],
            process=Process.sequential,
            verbose=1,
        )

        result = crew.kickoff()
        logger.info(f"Crew execution result: {result}")

    except Exception as e:
        logger.error(f"Error during crew execution: {e}")

if __name__ == "__main__":
    main()
