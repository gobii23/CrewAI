from crewai import Agent, LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="gemini/gemini-2.5-flash")


class AutoMLAgents:
    def ml_engineer_agent(self):
        return Agent(
            role="Senior Machine Learning Engineer",
            goal="""Design, train, and optimize machine learning models using best practices
            for model selection, preprocessing, feature handling, train/test splits,
            cross-validation, and evaluation metrics. Ensure all steps are reproducible
            and artifacts are saved for audit and reuse.""",
            backstory="""You are a highly experienced ML engineer with expertise in PyCaret
            and automated ML pipelines. You are proficient in handling preprocessing,
            categorical encoding, missing value imputation, feature scaling, outlier
            management, model selection, hyperparameter tuning, and model evaluation.
            You ensure that every step is documented, reproducible, and saved to disk.
            You focus on building robust, interpretable, and deployable ML pipelines.""",
            verbose=True,
            llm=llm,
        )
