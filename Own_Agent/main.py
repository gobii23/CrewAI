from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv
from crewai_tools import SerperDevTool
from tools.plot_tool import PlotTool
from tools.semantic_CSVsearch import AutoSemanticCSVTool
load_dotenv()

search_tool = SerperDevTool()

llm = LLM(
    model="gemini/gemini-2.5-flash"
)

query = input("Enter your query: ")


gopinath_agent = Agent(
    role="Senior Artificial Intelligence and Machine Learning Engineer",
    goal=("""Design, develop, and deploy advanced AI/ML systems that are scalable, efficient, and aligned with the crew's mission.
        Ensure all models are well-validated, interpretable, and integrated into resilient pipelines.
        Continuously drive technical innovation by evaluating cutting-edge algorithms and tools, translating them into reliable production-ready solutions that maximize
        autonomy, learning efficiency, and measurable outcomes."""),

    backstory=("""With over a decade of experience building production-grade machine learning systems, 
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

data_visualization_task = Task(
    description=(f"""Analyze the provided CSV file to generate a bar chart that answers the user's question.
        Instructions:
        1. Understand the dataset structure and interpret its contents accurately. Stick strictly to the data.
        2. Use the AutoSemanticCSVTool to explore and extract relevant insights.
        3. Use the PlotTool for creating visual plots.
        4. You must only return a plot image as the final output, no explanations or text.
        Input Format for PlotTool should be a JSON string like: {{\"labels\": [...], \"values\": [...], \"title\": ..., \"xlabel\": ..., \"ylabel\": ...}}
        User Question:{query}"""),
    expected_output="Only a bar chart image that visually answers the user's question based on the CSV data."
        "No additional text or explanation should be included in the final output",
    tools=[AutoSemanticCSVTool(csv_path="IPL.csv"), PlotTool()],
    agent=gopinath_agent
)


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

        Go beyond superficial mentions—aim to uncover **why** each development matters, how it's influencing the industry, and what its long-term implications might be.
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
    output_file="summary.md",
    agent=gopinath_agent
)

crew = Crew(
    agents=[gopinath_agent],
    tasks=[data_visualization_task],
    verbose=True
)

result = crew.kickoff()
