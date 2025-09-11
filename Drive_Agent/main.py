import streamlit as st
from crewai import Agent, Task, Crew, LLM
from drive_notification_tool import DriveWatcherTool
from drive_tree_tool import DriveTreeTool
from dotenv import load_dotenv
import os
import pandas as pd  

load_dotenv()

# --- LLM ---
llm = LLM(model="gemini/gemini-2.5-flash")

# --- Tools ---
drive_watcher_tool = DriveWatcherTool()
drive_tree_tool = DriveTreeTool()

# --- Agents ---
drive_tree_agent = Agent(
    role="Google Drive Explorer",
    goal="Explore and map the directory tree of Google Drive folders.",
    backstory="You map Google Drive folders into structured tree views.",
    llm=llm,
    verbose=True,
)

drive_watcher_agent = Agent(
    role="Google Drive Monitor",
    goal="Continuously watch a Google Drive folder for changes and print real-time notifications.",
    backstory="You monitor shared Google Drive folders and report changes instantly.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# --- Streamli UI ---
st.title("Google Drive Directory Monitor")

folder_input = st.text_input("Enter the Google Drive Folder URL or ID:")

if st.button("Run Tasks") and folder_input:
    # Delete old files 
    for file in ["Directory_Tree.md", "DriveWatcher.md"]:
        if os.path.exists(file):
            os.remove(file)

    # --- Tasks ---
    drive_tree_task = Task(
        description=f"Generate directory tree for {folder_input} and save to Directory_Tree.md",
        expected_output="Markdown file with directory tree",
        tools=[drive_tree_tool],
        agent=drive_tree_agent,
    )

    drive_watch_task = Task(
        description=f"Monitor folder {folder_input} and save events to DriveWatcher.md",
        expected_output="Markdown file with logs",
        tools=[drive_watcher_tool],
        agent=drive_watcher_agent,
    )

    crew = Crew(
        agents=[drive_tree_agent, drive_watcher_agent],
        tasks=[drive_tree_task, drive_watch_task],
        verbose=True
    )

    with st.spinner("Running tasks..."):
        crew.kickoff()

    st.success("Both tasks completed! Expand below to view results.")

    # --- Dropdowns for Markdown files ---
    if os.path.exists("Directory_Tree.md"):
        with st.expander("View Directory Tree"):
            with open("Directory_Tree.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())

    if os.path.exists("DriveWatcher.md"):
        with st.expander("View Drive Updates"):
            with open("DriveWatcher.md", "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Convert Markdown table into DataFrame
            if len(lines) > 2:
                header = [h.strip() for h in lines[0].strip().split("|")[1:-1]]
                rows = [line.strip().split("|")[1:-1] for line in lines[2:]]
                df = pd.DataFrame(rows, columns=header)

                st.table(df)
            else:
                st.info("No updates yet.")
 
