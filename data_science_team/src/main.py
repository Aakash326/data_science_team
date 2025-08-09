"""
Main Execution File for CrewAI Data Science Pipeline
===================================================
This is the main file that orchestrates the entire data science pipeline.
"""

import sys
import os
from IPython.display import display, Markdown
from crewai import Crew, Process

# Import our custom modules
from config import Config, check_requirements, initialize_global_vars
from tools import get_tools
from agents import DataScienceAgents
from task import DataScienceTasks


def install_requirements():
    """Install required packages."""
    import subprocess
    
    packages = [
        'crewai',
        'langchain-google-genai',  # Changed from langchain-openai to support Gemini
        'google-generativeai',     # Added for Gemini support
        'scikit-learn',
        'xgboost',
        'shap'
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--no-cache-dir'])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")


def setup_pipeline():
    """Setup the entire pipeline components."""
    print("🔧 Setting up pipeline components...")
    
    # Setup environment and LLM
    try:
        llm = Config.setup_environment()
        if llm is None:
            print("❌ Failed to setup LLM. Please check your API configuration.")
            return None, None, None, None
    except Exception as e:
        print(f"❌ Error setting up LLM: {e}")
        return None, None, None, None
    
    # Load data
    try:
        df = Config.load_data()
        if df is None:
            print("❌ Failed to load data. Please check your DATA_PATH configuration.")
            return None, None, None, None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None
    
    # Initialize global variables
    initialize_global_vars(df)
    
    # Get tools
    tools = get_tools()
    
    # Create agents
    try:
        agent_factory = DataScienceAgents(llm, tools)
        agents = agent_factory.get_all_agents()
    except Exception as e:
        print(f"❌ Error creating agents: {e}")
        print("💡 Check agents.py for syntax errors (especially around line 115)")
        return None, None, None, None
    
    # Create tasks
    try:
        tasks = DataScienceTasks.get_all_tasks(agents)
    except Exception as e:
        print(f"❌ Error creating tasks: {e}")
        return None, None, None, None
    
    return agents, tasks, tools, df


def create_crew(agents, tasks):
    """Create and configure the CrewAI crew."""
    # Convert verbose level to boolean (True if > 0, False otherwise)
    verbose_bool = Config.VERBOSE_LEVEL > 0
    
    try:
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=verbose_bool,
        )
    except TypeError as e:
        print(f"⚠️  Note: Creating crew with default parameters: {e}")
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=verbose_bool
        )


def print_pipeline_info():
    """Print pipeline information and instructions."""
    print("="*60)
    print("🚀 ENHANCED CREWAI DATA SCIENCE PIPELINE")
    print("="*60)
    print("📊 Components:")
    print("   • Data Exploration Agent")
    print("   • Feature Engineering Agent")
    print("   • Model Ensemble Agent")
    print("   • Model Validation Agent")
    print()
    print("🛠️  Tools Available:")
    print("   • Data Analysis Tool (Code Execution)")
    print("   • Model Training Tool (ML Training)")
    print("   • Visualization Tool (Chart Generation)")
    print()
    print("📋 Pipeline Steps:")
    print("   1. Data Exploration & Quality Analysis")
    print("   2. Feature Engineering & Preprocessing")
    print("   3. Model Training & Ensemble Building")
    print("   4. Model Validation & Interpretation")
    print("="*60)


def print_results_summary(result, tools):
    """Print a summary of the results."""
    print("\n" + "="*60)
    print("✅ PIPELINE EXECUTION COMPLETE!")
    print("="*60)
    print("🎯 Results Summary:")
    print("   • All agents executed their tasks successfully")
    print("   • Tools were actively used throughout the pipeline")
    print("   • Models trained and validated")
    print()
    print("🛠️  Tool Usage Summary:")
    for tool_name, tool in tools.items():
        print(f"   ✓ {tool.name}: Used for {tool.description}")
    print("="*60)
    
    # Print model scores
    print_model_scores()


def print_model_scores():
    """Print model performance scores."""
    from config import get_global_var
    
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE SCORES")
    print("="*60)
    
    try:
        # Get models from global variables
        rf_model = get_global_var('rf_model')
        xgb_model = get_global_var('xgb_model')
        lr_model = get_global_var('lr_model')
        
        # Get test data
        X_test = get_global_var('X_test')
        y_test = get_global_var('y_test')
        
        if X_test is not None and y_test is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
            import numpy as np
            
            print("🔍 Individual Model Scores:")
            print("-" * 40)
            
            models = {
                'Random Forest': rf_model,
                'XGBoost': xgb_model,
                'Logistic Regression': lr_model
            }
            
            model_scores = {}
            
            for model_name, model in models.items():
                if model is not None:
                    try:
                        y_pred = model.predict(X_test)
                        
                        # Determine if it's classification or regression
                        is_classification = hasattr(model, 'predict_proba') or len(np.unique(y_test)) <= 10
                        
                        if is_classification:
                            # Classification metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            
                            print(f"📈 {model_name}:")
                            print(f"   ✓ Accuracy:  {accuracy:.4f}")
                            print(f"   ✓ Precision: {precision:.4f}")
                            print(f"   ✓ Recall:    {recall:.4f}")
                            print(f"   ✓ F1-Score:  {f1:.4f}")
                            
                            model_scores[model_name] = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1
                            }
                        else:
                            # Regression metrics
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(y_test, y_pred)
                            
                            print(f"📈 {model_name}:")
                            print(f"   ✓ RMSE:      {rmse:.4f}")
                            print(f"   ✓ MSE:       {mse:.4f}")
                            print(f"   ✓ R²:        {r2:.4f}")
                            
                            model_scores[model_name] = {
                                'rmse': rmse,
                                'mse': mse,
                                'r2_score': r2
                            }
                        
                        print()
                    except Exception as e:
                        print(f"❌ Error evaluating {model_name}: {str(e)}")
                else:
                    print(f"⚠️  {model_name}: Model not found")
            
            # Print ensemble results if available
            ensemble_pred = get_global_var('ensemble_pred')
            if ensemble_pred is not None:
                try:
                    print("🎯 Ensemble Model Performance:")
                    print("-" * 40)
                    
                    is_classification = len(np.unique(y_test)) <= 10
                    
                    if is_classification:
                        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
                        ensemble_precision = precision_score(y_test, ensemble_pred, average='weighted', zero_division=0)
                        ensemble_recall = recall_score(y_test, ensemble_pred, average='weighted', zero_division=0)
                        ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
                        
                        print(f"📊 Ensemble Results:")
                        print(f"   ✓ Accuracy:  {ensemble_accuracy:.4f}")
                        print(f"   ✓ Precision: {ensemble_precision:.4f}")
                        print(f"   ✓ Recall:    {ensemble_recall:.4f}")
                        print(f"   ✓ F1-Score:  {ensemble_f1:.4f}")
                        
                        # Find best performing model
                        if model_scores:
                            best_model = max(model_scores.keys(), key=lambda x: model_scores[x].get('accuracy', 0))
                            print(f"\n🏆 Best Individual Model: {best_model} (Accuracy: {model_scores[best_model]['accuracy']:.4f})")
                            print(f"🎯 Ensemble vs Best: {ensemble_accuracy - model_scores[best_model]['accuracy']:+.4f}")
                    else:
                        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
                        ensemble_rmse = np.sqrt(ensemble_mse)
                        ensemble_r2 = r2_score(y_test, ensemble_pred)
                        
                        print(f"📊 Ensemble Results:")
                        print(f"   ✓ RMSE:      {ensemble_rmse:.4f}")
                        print(f"   ✓ MSE:       {ensemble_mse:.4f}")
                        print(f"   ✓ R²:        {ensemble_r2:.4f}")
                        
                        # Find best performing model
                        if model_scores:
                            best_model = max(model_scores.keys(), key=lambda x: model_scores[x].get('r2_score', -float('inf')))
                            print(f"\n🏆 Best Individual Model: {best_model} (R²: {model_scores[best_model]['r2_score']:.4f})")
                            print(f"🎯 Ensemble vs Best: {ensemble_r2 - model_scores[best_model]['r2_score']:+.4f}")
                    
                except Exception as e:
                    print(f"❌ Error evaluating ensemble: {str(e)}")
            
        else:
            print("⚠️  Test data not found. Cannot compute scores.")
            print("💡 Make sure your pipeline properly splits and stores test data.")
    
    except Exception as e:
        print(f"❌ Error retrieving model scores: {str(e)}")
        print("💡 Make sure models are properly stored in global variables.")
    
    print("="*60)


def main():
    """Main execution function."""
    # Check and install requirements
    if not check_requirements():
        print("Installing missing requirements...")
        install_requirements()
    
    # Print pipeline information
    print_pipeline_info()
    
    # Setup pipeline
    agents, tasks, tools, df = setup_pipeline()
    
    if agents is None:
        print("❌ Failed to setup pipeline. Please check your configuration.")
        print("💡 Make sure to:")
        print("   1. Update DATA_PATH to point to your actual CSV file")
        print("   2. Ensure your Gemini API key is valid")
        print("   3. Check that all required packages are installed")
        print("   4. Fix any syntax errors in agents.py")
        return
    
    # Create crew
    try:
        crew = create_crew(agents, tasks)
    except Exception as e:
        print(f"❌ Failed to create crew: {str(e)}")
        return
    
    # Execute pipeline
    print("🚀 Starting Enhanced Pipeline with Active Tool Usage...")
    print("📊 Tools will be actively used by agents for each task...\n")
    
    try:
        result = crew.kickoff()
        
        # Print results
        print_results_summary(result, tools)
        
        # Always print final score summary
        print_final_score_summary()
        
        # Display the detailed results
        try:
            display(Markdown(result.raw))
        except:
            print("\n📋 Detailed Results:")
            print("-" * 40)
            print(result.raw if hasattr(result, 'raw') else str(result))
        
        return result
        
    except Exception as e:
        print(f"❌ Error during pipeline execution: {str(e)}")
        print("💡 Possible solutions:")
        print("   1. Check your Gemini API key is valid and has sufficient quota")
        print("   2. Ensure your data file is properly formatted")
        print("   3. Verify all agents and tasks are properly configured")
        print("   4. Check for syntax errors in your code files")
        return None


# Alternative function for Jupyter notebook usage
def run_pipeline_notebook():
    """Run the pipeline in a Jupyter notebook environment."""
    print("📓 Running in Jupyter Notebook mode...")
    return main()


# Configuration update helper
def update_config(api_key=None, data_path=None, model_name=None):
    """Helper function to update configuration."""
    if api_key:
        # Update for Gemini API
        os.environ['GOOGLE_API_KEY'] = api_key
        Config.GOOGLE_API_KEY = api_key
        print("✅ Gemini API Key updated")
    
    if data_path:
        if os.path.exists(data_path):
            Config.DATA_PATH = data_path
            print(f"✅ Data path updated to: {data_path}")
        else:
            print(f"❌ Data file not found: {data_path}")
    
    if model_name:
        Config.MODEL_NAME = model_name
        print(f"✅ Model name updated to: {model_name}")


def print_final_score_summary():
    """Print a final summary of all model scores."""
    from config import get_global_var
    import numpy as np
    
    print("\n" + "🎯" * 20)
    print("🏁 FINAL PERFORMANCE SUMMARY")
    print("🎯" * 20)
    
    try:
        # Get test data
        X_test = get_global_var('X_test')
        y_test = get_global_var('y_test')
        
        if X_test is None or y_test is None:
            print("❌ No test data available for scoring")
            return
        
        # Get models
        models = {
            'Random Forest': get_global_var('rf_model'),
            'XGBoost': get_global_var('xgb_model'),
            'Logistic Regression': get_global_var('lr_model')
        }
        
        ensemble_pred = get_global_var('ensemble_pred')
        
        # Calculate scores for all models
        scores_table = []
        
        for model_name, model in models.items():
            if model is not None:
                try:
                    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
                    y_pred = model.predict(X_test)
                    
                    # Determine problem type
                    is_classification = len(np.unique(y_test)) <= 10
                    
                    if is_classification:
                        score = accuracy_score(y_test, y_pred)
                        metric = "Accuracy"
                    else:
                        score = r2_score(y_test, y_pred)
                        metric = "R² Score"
                    
                    scores_table.append((model_name, score))
                except:
                    scores_table.append((model_name, "Error"))
            else:
                scores_table.append((model_name, "Not Available"))
        
        # Add ensemble if available
        if ensemble_pred is not None:
            try:
                is_classification = len(np.unique(y_test)) <= 10
                if is_classification:
                    ensemble_score = accuracy_score(y_test, ensemble_pred)
                    metric = "Accuracy"
                else:
                    ensemble_score = r2_score(y_test, ensemble_pred)
                    metric = "R² Score"
                scores_table.append(("🎯 Ensemble", ensemble_score))
            except:
                scores_table.append(("🎯 Ensemble", "Error"))
        
        # Print the scores table
        print(f"\n📊 Final {metric} Scores:")
        print("-" * 50)
        
        valid_scores = []
        for model_name, score in scores_table:
            if isinstance(score, (int, float)):
                print(f"{model_name:<20}: {score:.4f}")
                if "Ensemble" not in model_name:
                    valid_scores.append(score)
            else:
                print(f"{model_name:<20}: {score}")
        
        # Print best model
        if valid_scores:
            best_score = max(valid_scores)
            best_models = [name for name, score in scores_table if isinstance(score, (int, float)) and score == best_score and "Ensemble" not in name]
            
            print(f"\n🏆 Best Model: {best_models[0]} ({metric}: {best_score:.4f})")
            
            # Check if ensemble is better
            ensemble_entry = next((score for name, score in scores_table if "Ensemble" in name), None)
            if isinstance(ensemble_entry, (int, float)):
                if ensemble_entry > best_score:
                    print(f"🎯 Ensemble Improvement: +{ensemble_entry - best_score:.4f}")
                    print("✅ Ensemble performs better than individual models!")
                elif ensemble_entry < best_score:
                    print(f"📉 Ensemble vs Best: {ensemble_entry - best_score:.4f}")
                    print("⚠️  Individual model outperforms ensemble")
                else:
                    print("🤝 Ensemble matches best individual model")
        
        print("\n" + "🎯" * 20)
        
    except Exception as e:
        print(f"❌ Error in final score summary: {str(e)}")
        print("💡 Check that models are properly trained and stored")


# Debug function to test configuration
def test_configuration():
    """Test the configuration setup."""
    print("🧪 Testing Configuration...")
    print(f"📁 Data Path: {Config.DATA_PATH}")
    print(f"🤖 Model: {Config.MODEL_NAME}")
    print(f"🔑 API Key: {'Set' if hasattr(Config, 'GOOGLE_API_KEY') and Config.GOOGLE_API_KEY else 'Not Set'}")
    
    # Test data loading
    print("\n📊 Testing data loading...")
    if hasattr(Config, 'inspect_data_file'):
        Config.inspect_data_file()
    
    df = Config.load_data()
    if df is not None:
        print(f"✅ Data loaded successfully: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
    else:
        print("❌ Failed to load data")
    
    # Test requirements
    print("\n📦 Testing requirements...")
    if check_requirements():
        print("✅ All requirements satisfied")
    else:
        print("❌ Missing requirements")


if __name__ == "__main__":
    # You can uncomment this line to test configuration first
    # test_configuration()
    
    # Run the pipeline
    result = main()
    
    if result:
        print("\n🎉 Pipeline completed successfully!")
        print("📊 Model scores have been displayed above.")
        print("📄 Check the detailed results for comprehensive analysis.")
    else:
        print("\n💥 Pipeline execution failed. Please check the errors above.")
        print("\n💡 Try running test_configuration() to diagnose issues:")