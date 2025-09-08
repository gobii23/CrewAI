from crewai import Agent, Task, Crew, LLM
from drive_notification_tool import DriveWatcherTool
from drive_tree_tool import DriveTreeTool
from dotenv import load_dotenv

load_dotenv()

# --- Initialize LLM ---
llm = LLM(
    model="gemini/gemini-2.5-flash"
)

# --- Initialize Tools ---
drive_watcher_tool = DriveWatcherTool()
drive_tree_tool = DriveTreeTool()

# --- Initialize Agent ---
drive_tree_agent = Agent(
    role="Google Drive Explorer",
    goal=(
        "Explore and map the directory tree of Google Drive folders. "
        "Present the results in a clear, readable hierarchy."
    ),
    backstory=(
        "You are a meticulous digital cartographer who loves mapping file systems. "
        "Your specialty is traversing Google Drive folders and providing users with "
        "a structured tree view of all their files and subfolders."
    ),
    llm=llm,
    verbose=True,
)

drive_watcher_agent = Agent(
    role="Google Drive Monitor",
    goal=(
        "Continuously watch a Google Drive folder for changes "
        "and send email updates when new files are added, modified, or deleted."
    ),
    backstory=(
        "You are a diligent assistant who keeps track of everything happening in "
        "a shared Google Drive folder. You send timely email updates to the team "
        "so no one misses important changes."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# --- Get User Inputs ---
folder_input = input("Enter the Google Drive Folder URL or ID: ")
recipient_email = input("Enter the recipient email: ")

# --- Define Task ---
drive_tree_task = Task(
    description=(
        f"""Use the DriveTreeTool to generate a directory tree of the google drive folder '{folder_input}' "
        "by the user. Ensure that the output is formatted in a clean hierarchy "
        "with 📂 for folders and 📄 for files.\n\n"
        "Your final answer MUST be the full directory tree as a plain text block "
        "that can be easily read by humans."""
    ),
    expected_output="A complete directory tree of the target folder, formatted with indentation.",
    tools=[drive_tree_tool],
    agent=drive_tree_agent,
)

drive_watch_task = Task(
    description=(
        f"Start monitoring the Google Drive folder '{folder_input}'. "
        f"Send updates to '{recipient_email}' every time files are added, modified, or deleted. "
        "You must run continuously until stopped."
    ),
    expected_output=(
        "The folder is being watched continuously. "
        "Emails with detected changes are sent to the specified recipient."
    ),
    tools=[drive_watcher_tool],
    agent=drive_watcher_agent
)

# --- Initialize Crew ---
crew = Crew(
    agents=[drive_tree_agent, drive_watcher_agent],
    tasks=[drive_tree_task, drive_watch_task],
    verbose=True
)

# --- Start the Crew ---
crew.kickoff()
