from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv
import re
import os
import json
import pandas as pd
from crewai_tools import SerperDevTool
from tools.plot_tool import PlotTool
from tools.eda_tool import EDATool
from tools.csv_rag_tool import CsvRAGTool
from tools.terminal_tool import TerminalTool
from tools.data_preprocessing_tool import DataPreprocessingTool

load_dotenv()

# LLM MODEL
llm = LLM(model="gemini/gemini-2.5-flash")


# TOOLS
search_tool = SerperDevTool()
terminal_tool = TerminalTool()
eda_tool = EDATool()
rag_tool = CsvRAGTool(file_path="data/csv_cleaned_data/titanic_data_cleaned.csv")
plot_tool = PlotTool()
preprocessing_tool = DataPreprocessingTool()


# INPUT
query = input("Enter your query: ")
data_path = input("Enter a name of csv: ")
file_path = f"data/csv_data/{data_path}.csv"
csv_path = (f"data/csv_cleaned_data/{data_path}_cleaned.csv")
try:
    df = pd.read_csv(file_path, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
print("\nColumns in the dataset:")
print(", ".join(df.columns.tolist()))
columns = input("\nEnter the columns to preprocess (comma-separated): ")
selected_columns = [
    col.strip() for col in columns.split(",") if col.strip() in df.columns
]
if not selected_columns:
    print("No valid columns selected. Exiting.")


# FILENAME
safe_filename = re.sub(r"[^a-zA-Z0-9_\-]", "_", query.strip().lower())[:50]
summary_filename = f"output/summary/summary_{safe_filename}.md"
os.makedirs("output/summary", exist_ok=True)


# AGENT
gopinath_agent = Agent(
    role="Senior Artificial Intelligence and Machine Learning Engineer",
    goal=(
        """Design, develop, and deploy advanced AI/ML systems that are scalable, efficient, and aligned with the crew's mission.
        Ensure all models are well-validated, interpretable, and integrated into resilient pipelines.
        Continuously drive technical innovation by evaluating cutting-edge algorithms and tools, translating them into reliable production-ready solutions that maximize
        autonomy, learning efficiency, and measurable outcomes."""
    ),
    backstory=(
        """With over a decade of experience building production-grade machine learning systems, 
        you specialize in turning cutting-edge AI research into scalable, maintainable, and high-performing software solutions. 
        You've led cross-functional engineering teams in designing robust ML pipelines, deploying real-time models, and setting up performance monitoring infrastructure at scale.
        In addition, your strength lies in end-to-end data handling — from ingestion and cleansing, to transformation, 
        feature engineering, and large-scale distributed processing. You bring deep expertise in exploratory data analysis (EDA), 
        statistical validation, and designing insightful data visualizations that uncover patterns and guide business strategy.
        Your academic foundation in mathematics and computer science is complemented by industry leadership in MLOps, 
        data governance, and cloud-native AI systems. You've contributed to everything from fraud detection in fintech 
        to recommender engines in global e-commerce, always maintaining a rigorous focus on reproducibility, explainability, 
        and measurable business impact.
        You are well known for driving engineering excellence and ethical AI adoption, with a track record of mentoring teams, 
        validating models against relevant KPIs, and establishing best practices in experimentation, system architecture, and responsible AI deployment."""
    ),
    inject_date=True,
    llm=llm,
    verbose=True,
)

# TASKS
# TASK_1
data_visualization_task = Task(
    description=(
        f"""Analyze the provided CSV file to generate a bar chart that answers the user's question.
        Instructions:
        1. Understand the dataset structure and interpret its contents accurately. Stick strictly to the data.
        2. Use the csv_tool to convert the csv into a Vector DB and then use rag_tool to explore and extract relevant insights.
        3. Use the plot_tool for creating visual plots.
        4. You must only return a plot image as the final output, no explanations or text.
        Input Format for plot_tool should be a JSON string like: {{\"labels\": [...], \"values\": [...], \"title\": ..., \"xlabel\": ..., \"ylabel\": ...}}
        User Question:{query}"""
    ),
    expected_output="Only a bar chart image that visually answers the user's question based on the CSV data."
    "No additional text or explanation should be included in the final output",
    tools=[rag_tool, plot_tool],
    human_input=True,
    agent=gopinath_agent,
)


# TASK_2
research_innovation_task = Task(
    description=(
        f"""
        Conduct a thorough investigation into the latest trends and innovations in the {query}.
        Search across reputable sources including recent news articles, white papers, industry blogs,
        press releases, and academic publications. Focus on content published within the last 6–12 months.

        Your goal is to synthesize findings into an insightful and well-organized report that covers:
        1.Breakthrough innovations or product launches disrupting the {query} space.
        2.Strategic initiatives, partnerships, or acquisitions by key industry players.
        3.Emerging market demands, evolving consumer behaviors, or demographic shifts.
        4.Technological advancements, R&D efforts, or scientific milestones.
        5.Regulatory changes, policy updates, or geopolitical influences impacting the field.

        Go beyond superficial mentions—aim to uncover 'why' each development matters, how it's influencing the industry, and what its long-term implications might be.
        Maintain high factual accuracy. Only include information from reliable sources and always cite them.
        Cross-verify findings where possible to ensure credibility."""
    ),
    expected_output=f"""A detailed report (with bullet points) highlighting most critical insights related to innovations and trends in the {query}.
    Output MUST include:
        1.A short executive summary (2–3 sentences) describing the overall innovation landscape.
        2.A bullet-pointed list of key findings with the following structure for each point:
          a. Topic/Headline: Short and descriptive title
          b. Insight: 1–2 sentence explanation of why this trend or event is significant
          c. Source: Cite the exact URL or publication (with date if possible)
    Your final output MUST be written in Markdown syntax and formatted cleanly for readability (e.g., use headings, subheadings, bullet points, and links).""",
    tools=[search_tool],
    output_file=summary_filename,
    agent=gopinath_agent,
)

# TASK_3
data_collection_Ingestion_task = Task(
    description=f"""
    1. Use a web search (e.g., Google) restricted to site:kaggle.com to find the most relevant and recent Kaggle dataset for the user’s query: {query}. Prioritize:
        a. Datasets closely aligned in topic/content.
        b. Recently updated or published datasets (e.g., within the last 12 months)—give preference to newer ones if multiple are similarly relevant.
    2. From the dataset’s Kaggle URL, extract the dataset slug in the form: ownername/dataset-name.
    3. Using the Kaggle CLI, perform the following steps:
        a. Download the dataset:
            kaggle datasets download -d <slug> -p ./data --unzip=false
        b. Extract the contents into `./data`.
        c. Delete all non-CSV files in `./data`, preserving only `.csv` files.
        d. Confirm (e.g., via listing) that only CSV files remain.
    Implementation notes:
    1. To extract the slug from a Kaggle dataset URL like `https://www.kaggle.com/datasets/ownername/dataset-name`, use a regex such as: `https?://www\.kaggle\.com/datasets/([^/]+/[^/?#]+)`.
    2. After download, you can unzip and clean with a shell snippet like:

    ```bash
    # Download
    kaggle datasets download -d "$SLUG" -p ./data

    # Unzip and remove originals
    unzip -o ./data/"${{SLUG##*/}}".zip -d ./data
    rm ./data/*.zip

    # Remove non-CSV files
    find ./data -type f ! -name "*.csv" -delete

    # Confirm only CSVs remain
    ls -1 ./data

    If no recent dataset exists, fall back to the most relevant one regardless of age, but indicate its last update date.""",
    expected_output="""The dataset has been successfully downloaded and any CSV files it contains are placed in the "./data/csv_data" directory.""",
    tools=[search_tool, terminal_tool],
    agent=gopinath_agent,
)


#TASK_4
data_preprocessing_task = Task(
    description=f"""The task is preprocessing a dataset loaded from the file path: {file_path}.
        Objective:
            Carefully analyze the dataset and guide the user through selecting specific columns for preprocessing. Your job is to process only those selected columns {selected_columns} while preserving all other columns as-is in the final output.
        Instructions:
            1. Initial Data Inspection:
                a. Load the CSV from the given file path.
                b. List all columns in the dataset to help the user make an informed decision.
            2. User Interaction:
                a. Prompt the user to manually input the columns they want to preprocess.
                b. Validate that the selected columns exist in the dataset. If invalid columns are selected, notify and abort preprocessing.
            3. Preprocessing Logic (only for selected columns : {selected_columns} ):
                Handling Missing Values:
                    a. For numerical columns, fill missing values using either:
                    b. Mean, if data is symmetrically distributed.
                    c. Median, if the column is skewed.
                    d. Drop rows that have too many missing values to be reliably imputed.
                Encoding Categorical Columns:
                    a. Identify whether a column is categorical.
                    b. If it is a binary categorical column (2 unique values), apply Label Encoding.
                    c. If it is a multi-class categorical column with 3–20 unique values, apply One-Hot Encoding.
                    d. If it has more than 20 unique values, consider it high cardinality and **drop the column** to prevent noise and sparsity.
            4. Data Consolidation:
                a. After preprocessing the selected columns, merge them back with the untouched columns.
                b. Ensure the order of rows and columns is maintained.
            5. Save the Output:
                a. Save the final cleaned dataset in CSV format.
                b. The file should be placed inside the directory: `./data/csv_data/csv_cleaned_data/`.
                c. Name the file using the convention: `<original_filename>_cleaned.csv`.
            Notes:
                a. This process must be interactive. Ask the user for input during execution.
                b. Do not process any columns that the user did not select.
                c. Ensure the final dataset is suitable for machine learning pipelines — no nulls in processed columns, and encoding is done cleanly.""",
    expected_output="The user-selected columns are cleaned and encoded, and the resulting dataset is saved as a CSV in ./data/csv_data/csv_cleaned_data/.",
    tools=[preprocessing_tool],
    agent=gopinath_agent,
)

#TASK_5
exploratory_data_analysis = Task(
    description=(
        f"""Perform a thorough Exploratory Data Analysis (EDA) on the dataset loaded from {csv_path} "
        "You must generate charts that help in understanding the data distribution, outliers, and correlations.\n\n"
        "Specifically, your job is to:\n"
        "1. Generate histograms and boxplots for all numerical columns to analyze distributions and outliers.\n"
        "2. Generate bar charts for all categorical columns to understand frequency distributions.\n"
        "3. Create a correlation heatmap to show relationships between numerical variables.\n"
        "4. Save all the plots in a structured folder.\n"
        "5. Return the output directory and total number of charts generated.\n\n"
        "Make sure the CSV file path is valid and handle errors gracefully."""
    ),
    expected_output=(
        "A JSON string with the following structure:\n"
        "{{\n"
        "  'status': 'success',\n"
        "  'output_dir': '<path_to_output_directory>',\n"
        "  'charts_generated': <total_number_of_charts>\n"
        "}}\n\n"
        "All charts should be saved as PNG files in the specified output folder."
    ),
    tools=[rag_tool, eda_tool],
    agent=gopinath_agent
)


# CREW
crew = Crew(
    agents=[gopinath_agent],
    tasks=[research_innovation_task],
    verbose=True
)

# KICKOFF
result = crew.kickoff()
