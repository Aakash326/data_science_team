"""
Agent Definitions for CrewAI Data Science Pipeline
=================================================
This module contains all agent definitions with their roles, goals, and backstories.
"""

from crewai import Agent
# from crewai.tools import Tool
from langchain_openai import ChatOpenAI

# TEMP DUMMY TOOL for testing
class DummyTool:
    def run(self, code: str, description: str):
        print(f"Running tool...\n🧪 Description: {description}")
        return f"Executed: {description}"

class DataScienceAgents:
    """Factory class for creating data science agents."""
    
    def __init__(self, llm: ChatOpenAI, tools: dict):
        self.llm = llm
        self.tools = tools
    
    def create_data_exploration_agent(self):
        """Create the data exploration specialist agent."""
        return Agent(
            role="Senior Data Exploration Specialist",
            goal=(
                "Discover data insights and quality issues through autonomous analysis. "
                "Make independent decisions about what aspects of the data require attention."
            ),
            backstory=(
                "You are an experienced data scientist with deep intuition for data quality and patterns. "
                "You don't follow checklists - you investigate what matters most for each unique dataset. "
                "Your analysis guides all downstream decisions in the pipeline."
            ),
            llm=self.llm,
            tools=[self.tools['data_analysis_tool'], self.tools['visualization_tool']],
            allow_delegation=False,
            verbose=True
        )
    
    def create_feature_engineering_agent(self):
        """Create the feature engineering architect agent."""
        return Agent(
            role="Feature Engineering Architect",
            goal=(
                "Transform raw data into optimal features through intelligent decision-making. "
                "Autonomously determine what preprocessing and feature creation will maximize model performance."
            ),
            backstory=(
                "You are a feature engineering expert who understands that every dataset is unique. "
                "You make data-driven decisions about transformations, never following rigid templates. "
                "Your choices are based on deep analysis of data characteristics and modeling requirements."
            ),
            llm=self.llm,
            tools=[self.tools['data_analysis_tool']],
            allow_delegation=False,
            verbose=True
        )
    
    def create_model_ensemble_agent(self):
        """Create the ML ensemble specialist agent."""
        return Agent(
            role="ML Ensemble Specialist",
            goal=(
                "Design optimal modeling strategies tailored to the specific problem and data characteristics. "
                "Make intelligent choices about model types, optimization, and ensemble approaches."
            ),
            backstory=(
                "You are a machine learning expert who adapts methodology to each unique problem. "
                "You don't apply cookie-cutter approaches but instead choose techniques based on "
                "data properties, business requirements, and computational constraints."
            ),
            llm=self.llm,
            tools=[self.tools['model_training_tool'], self.tools['data_analysis_tool']],
            allow_delegation=False,
            verbose=True
        )
    
    def create_model_validation_agent(self):
        """Create the model validation and interpretation expert agent."""
        return Agent(
            role="Model Validation & Interpretation Expert",
            goal=(
                "Provide comprehensive model assessment and business-focused insights. "
                "Determine validation approaches that matter most for deployment success."
            ),
            backstory=(
                "You are a model validation expert who understands that different problems require "
                "different validation strategies. You focus on what stakeholders need to know to "
                "trust and effectively use models in real-world scenarios."
            ),
            llm=self.llm,
            tools=[self.tools['visualization_tool'], self.tools['data_analysis_tool']],
            allow_delegation=False,
            verbose=True
        )
    
    def get_all_agents(self):
        """Return all agents as a list."""
        return [
            self.create_data_exploration_agent(),
            self.create_feature_engineering_agent(),
            self.create_model_ensemble_agent(),
            self.create_model_validation_agent()
        ]


# Individual agent creation functions for more flexibility
def create_data_exploration_agent(llm, tools):
    """Create data exploration agent."""
    return Agent(
        role="Senior Data Exploration Specialist",
        goal=(
            "Discover data insights and quality issues through autonomous analysis. "
            "Make independent decisions about what aspects of the data require attention."
        ),
        backstory=(
            "You are an experienced data scientist with deep intuition for data quality and patterns. "
            "You don't follow checklists - you investigate what matters most for each unique dataset. "
            "Your analysis guides all downstream decisions in the pipeline."
        ),
        llm=llm,
        tools=[tools['data_analysis_tool'], tools['visualization_tool']],
        allow_delegation=False,
        verbose=True
    )


def create_feature_engineering_agent(llm, tools):
    """Create feature engineering agent."""
    return Agent(
        role="Feature Engineering Architect",
        goal=(
            "Transform raw data into optimal features through intelligent decision-making. "
            "Autonomously determine what preprocessing and feature creation will maximize model performance."
        ),
        backstory=(
            "You are a feature engineering expert who understands that every dataset is unique. "
            "You make data-driven decisions about transformations, never following rigid templates. "
            "Your choices are based on deep analysis of data characteristics and modeling requirements."
        ),
        llm=llm,
        tools=[tools['data_analysis_tool']],
        allow_delegation=False,
        verbose=True
    )


def create_model_ensemble_agent(llm, tools):
    """Create model ensemble agent."""
    return Agent(
        role="ML Ensemble Specialist",
        goal=(
            "Design optimal modeling strategies tailored to the specific problem and data characteristics. "
            "Make intelligent choices about model types, optimization, and ensemble approaches."
        ),
        backstory=(
            "You are a machine learning expert who adapts methodology to each unique problem. "
            "You don't apply cookie-cutter approaches but instead choose techniques based on "
            "data properties, business requirements, and computational constraints."
        ),
        llm=llm,
        tools=[tools['model_training_tool'], tools['data_analysis_tool']],
        allow_delegation=False,
        verbose=True
    )


def create_model_validation_agent(llm, tools):
    """Create model validation agent."""
    return Agent(
        role="Model Validation & Interpretation Expert",
        goal=(
            "Provide comprehensive model assessment and business-focused insights. "
            "Determine validation approaches that matter most for deployment success."
        ),
        backstory=(
            "You are a model validation expert who understands that different problems require "
            "different validation strategies. You focus on what stakeholders need to know to "
            "trust and effectively use models in real-world scenarios."
        ),
        llm=llm,
        tools=[tools['visualization_tool'], tools['data_analysis_tool']],
        allow_delegation=False,
        verbose=True
    )
if __name__ == "__main__":
    input_code = """
import pandas as pd
import numpy as np
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df.describe()
"""
    input_description = "Perform basic descriptive statistics using pandas."

    tool = DummyTool()
    output = tool.run(code=input_code, description=input_description)
    print("Output:\n", output)