"""
Robust Agent Definitions for CrewAI Data Science Pipeline
========================================================
Updated with better error handling and Gemini-specific optimizations.
"""

from crewai import Agent
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataScienceAgents:
    """Factory class for creating data science agents with robust error handling."""
    
    def __init__(self, llm, tools: dict):
        """
        Initialize the agent factory.
        
        Args:
            llm: Language model instance (Gemini ChatGoogleGenerativeAI)
            tools: Dictionary of available tools
        """
        self.llm = llm
        self.tools = tools
        logger.info(f"Agent factory initialized with LLM: {type(llm).__name__}")
    
    def create_data_exploration_agent(self):
        """Create the data exploration specialist agent with simplified instructions."""
        try:
            return Agent(
                role="Data Exploration Specialist",
                goal="Analyze the dataset to understand its structure, quality, and key patterns.",
                backstory=(
                    "You are a data scientist who specializes in exploratory data analysis. "
                    "You examine datasets systematically to understand their characteristics, "
                    "identify data quality issues, and discover important patterns that will "
                    "guide the modeling process."
                ),
                llm=self.llm,
                tools=[self.tools['data_analysis_tool'], self.tools['visualization_tool']],
                allow_delegation=False,
                verbose=True,
                max_execution_time=180,  # 3 minutes
                max_iter=3,  # Limit iterations
                memory=False  # Disable memory to reduce complexity
            )
        except Exception as e:
            logger.error(f"Failed to create data exploration agent: {e}")
            raise
    
    def create_feature_engineering_agent(self):
        """Create the feature engineering architect agent with simplified tasks."""
        try:
            return Agent(
                role="Feature Engineering Specialist", 
                goal="Prepare and transform the data for optimal machine learning performance.",
                backstory=(
                    "You are an expert in data preprocessing and feature engineering. "
                    "You know how to handle missing values, transform variables, and create "
                    "new features that will improve model performance. You make the data "
                    "ready for machine learning algorithms."
                ),
                llm=self.llm,
                tools=[self.tools['data_analysis_tool']],
                allow_delegation=False,
                verbose=True,
                max_execution_time=180,
                max_iter=3,
                memory=False
            )
        except Exception as e:
            logger.error(f"Failed to create feature engineering agent: {e}")
            raise
    
    def create_model_ensemble_agent(self):
        """Create the ML ensemble specialist agent with focused objectives."""
        try:
            return Agent(
                role="Machine Learning Specialist",
                goal="Train and optimize machine learning models for the best predictive performance.",
                backstory=(
                    "You are a machine learning expert who trains multiple models and "
                    "compares their performance. You understand different algorithms, "
                    "hyperparameter tuning, and ensemble methods. You select the best "
                    "approach for each specific dataset and problem."
                ),
                llm=self.llm,
                tools=[self.tools['model_training_tool'], self.tools['data_analysis_tool']],
                allow_delegation=False,
                verbose=True,
                max_execution_time=300,  # 5 minutes for training
                max_iter=3,
                memory=False
            )
        except Exception as e:
            logger.error(f"Failed to create model ensemble agent: {e}")
            raise
    
    def create_model_validation_agent(self):
        """Create the model validation specialist with clear evaluation focus."""
        try:
            return Agent(
                role="Model Validation Specialist",
                goal="Evaluate model performance and provide insights for business decision-making.",
                backstory=(
                    "You are a model validation expert who assesses how well models perform "
                    "and whether they are ready for deployment. You evaluate metrics, check "
                    "for biases, and provide clear recommendations about model reliability "
                    "and business impact."
                ),
                llm=self.llm,
                tools=[self.tools['visualization_tool'], self.tools['data_analysis_tool']],
                allow_delegation=False,
                verbose=True,
                max_execution_time=180,
                max_iter=3,
                memory=False
            )
        except Exception as e:
            logger.error(f"Failed to create model validation agent: {e}")
            raise
    
    def get_all_agents(self):
        """Return all agents as a list in the correct order with error handling."""
        try:
            logger.info("Creating all agents...")
            
            agents = [
                self.create_data_exploration_agent(),
                self.create_feature_engineering_agent(), 
                self.create_model_ensemble_agent(),
                self.create_model_validation_agent()
            ]
            
            logger.info(f"✅ Successfully created {len(agents)} agents:")
            for i, agent in enumerate(agents, 1):
                logger.info(f"   {i}. {agent.role}")
            
            return agents
            
        except Exception as e:
            logger.error(f"❌ Error creating agents: {str(e)}")
            raise


def test_agent_creation_with_mock():
    """Test agent creation with mock objects to verify structure."""
    print("🧪 Testing agent creation with mock objects...")
    
    # Mock LLM
    class MockLLM:
        def __init__(self):
            self.model_name = "gemini-1.5-flash"
            self.temperature = 0.1
        
        def invoke(self, message):
            return type('Response', (), {'content': 'Mock response'})()
    
    # Mock tools
    class MockTool:
        def __init__(self, name):
            self.name = name
            self.description = f"Mock {name}"
        
        def run(self, *args, **kwargs):
            return f"Mock execution of {self.name}"
    
    mock_tools = {
        'data_analysis_tool': MockTool('data_analysis_tool'),
        'model_training_tool': MockTool('model_training_tool'),
        'visualization_tool': MockTool('visualization_tool')
    }
    
    try:
        # Test agent factory
        mock_llm = MockLLM()
        agent_factory = DataScienceAgents(mock_llm, mock_tools)
        agents = agent_factory.get_all_agents()
        
        print(f"✅ Successfully created {len(agents)} agents with mock objects")
        
        # Test each agent
        for i, agent in enumerate(agents, 1):
            print(f"   {i}. {agent.role} - Tools: {len(agent.tools)}")
            
        return True
        
    except Exception as e:
        print(f"❌ Agent creation test failed: {str(e)}")
        return False


def create_minimal_test_pipeline():
    """Create a minimal test to isolate the LLM issue."""
    test_code = '''
# minimal_test.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def test_minimal_llm():
    """Test LLM with minimal configuration."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not set")
        return False
    
    try:
        # Create LLM with minimal settings
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            max_tokens=100
        )
        
        # Test simple invocation
        response = llm.invoke("Say 'Hello World'")
        print(f"✅ LLM Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_minimal_llm()
'''
    
    with open('minimal_test.py', 'w') as f:
        f.write(test_code)
    
    print("✅ Created minimal_test.py")
    print("Run with: python minimal_test.py")


if __name__ == "__main__":
    # Run all tests
    print("🔧 Testing Agent Structure...")
    test_agent_creation_with_mock()
    
    print("\n🔧 Creating Minimal LLM Test...")
    create_minimal_test_pipeline()
    
    print("\n💡 Next steps:")
    print("1. Run: python minimal_test.py")
    print("2. If that works, the issue is in CrewAI configuration")
    print("3. Try using gemini-1.5-flash instead of gemini-1.5-pro")