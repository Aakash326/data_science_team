"""
Task Definitions for CrewAI Data Science Pipeline
================================================
This module contains all task definitions for the data science workflow.
"""

from crewai import Task
import logging
from config import Config


def create_data_exploration_task(agent):
    """Create data exploration task."""
    real_path = Config.DATA_PATH
    return Task(
        description=(
            f"A CSV file located at `{real_path}` has already been loaded into a pandas DataFrame named `df`. "
            "Start your exploration with `print(df.head())` and `print(df.info())`.\n\n"
            "Analyze the dataset comprehensively to understand its structure, quality, and patterns:\n"
            "- Investigate data quality issues and assess what might need attention\n"
            "- Discover relationships between variables and identify patterns\n"
            "- Detect anomalies, outliers, or unusual data points\n"
            "- Understand the distribution and characteristics of each feature\n"
            "- Identify any temporal patterns if time-based data exists\n"
            "\nMake your own decisions about what aspects of the data are most important to explore. "
            "Use your available tools (data_analysis_tool for code execution, visualization_tool for charts) "
            "strategically based on what you discover. Let your analysis drive which tools you need."
        ),
        expected_output=(
            "Comprehensive analysis report with your findings, recommendations for data issues, "
            "and insights about feature relationships. Include visualizations that support your conclusions."
        ),
        agent=agent
    )


def create_feature_engineering_task(agent):
    """Create feature engineering task."""
    return Task(
        description=(
            "Based on the data exploration findings, engineer features to maximize model performance:\n"
            "- Decide which features to transform, create, or remove based on your analysis\n"
            "- Handle data quality issues using the most appropriate methods\n"
            "- Create new features that could improve predictive power\n"
            "- Apply preprocessing techniques where you see they're needed\n"
            "- Determine the optimal way to split the data for training and validation\n"
            "\nYou have full autonomy to make feature engineering decisions. Use your data_analysis_tool "
            "to execute the transformations you determine are necessary. Consider what the exploration "
            "agent found and make informed choices about data preparation."
        ),
        expected_output=(
            "Engineered dataset with documentation of your decisions: what you removed and why, "
            "what you created and the rationale, how you handled data issues, and final dataset characteristics."
        ),
        agent=agent
    )


def create_model_ensemble_task(agent):
    """Create model ensemble task."""
    return Task(
        description=(
            "Design and implement an optimal modeling strategy based on the prepared data:\n"
            "- Evaluate what types of models would work best for this problem\n"
            "- Determine appropriate hyperparameter optimization strategies\n"
            "- Decide on validation approaches that will give reliable performance estimates\n"
            "- Consider whether ensemble methods would add value\n"
            "- Choose evaluation metrics that align with the business problem\n"
            "\nYou have complete freedom to choose modeling approaches. Use your model_training_tool "
            "for automated model training and data_analysis_tool for custom analysis. Consider the nature "
            "of the data and problem to make informed decisions about model selection and optimization."
        ),
        expected_output=(
            "Trained models with justification for your choices: why you selected specific models, "
            "how you optimized them, what validation approach you used, and performance comparisons."
        ),
        agent=agent
    )


def create_model_validation_task(agent):
    """Create model validation task."""
    return Task(
        description=(
            "Comprehensively evaluate the models and provide actionable business insights:\n"
            "- Assess model reliability and generalization capability\n"
            "- Investigate model behavior and identify potential issues\n"
            "- Determine which factors drive model predictions\n"
            "- Evaluate business impact and practical deployment considerations\n"
            "- Identify scenarios where models might fail or need attention\n"
            "\nDecide what validation approaches are most important for this specific problem. "
            "Use your visualization_tool for charts and data_analysis_tool for statistical analysis. "
            "Your analysis should help stakeholders understand model trustworthiness and deployment readiness."
        ),
        expected_output=(
            "Complete validation assessment with model reliability analysis, interpretability insights, "
            "business recommendations, and clear guidance on model deployment and monitoring."
        ),
        agent=agent
    )


class DataScienceTasks:
    """Factory class for creating data science tasks."""
    
    @staticmethod
    def get_all_tasks(agents):
        """Create and return all tasks with their assigned agents."""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Defensive: Ensure agents list has at least 4 elements
        if len(agents) < 4:
            raise ValueError("At least 4 agents are required for the pipeline tasks.")
        
        try:
            tasks = []
            # Create tasks with proper error handling
            for i, (creator_func, agent) in enumerate([
                (create_data_exploration_task, agents[0]),
                (create_feature_engineering_task, agents[1]),
                (create_model_ensemble_task, agents[2]),
                (create_model_validation_task, agents[3])
            ]):
                try:
                    task = creator_func(agent)
                    tasks.append(task)
                    logger.info(f"Created task {i+1}: {task.description[:50]}...")
                except Exception as e:
                    logger.error(f"Failed to create task {i+1}: {str(e)}")
                    raise
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error in task creation pipeline: {str(e)}")
            raise