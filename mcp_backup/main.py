from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from dotenv import load_dotenv
import os
from agent import AutoMLAgents
from task import AutoMLTask
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

load_dotenv()

app = FastAPI(title="My Own Agent")


class MLRequest(BaseModel):
    dataset_path: str
    target_column: str
    task_type: str = "classification"
    output_path: str = "./output"
    dataset_name: Optional[str] = None


# MCP server parameters
server_params = StdioServerParameters(
    command="python",
    args=["server.py"],
    env={"UV_PYTHON": "3.11", **os.environ},
)


@app.get("/")
def read_root():
    return {"message": "My Own Agent", "status": "running"}


@app.post("/run_automl")
def run_automl(request: MLRequest):
    """Run complete AutoML pipeline using MCP tools"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            # Initialize agents and tasks
            agents = AutoMLAgents()
            automl_agent = agents.ml_engineer_agent()
            automl_agent.tools = list(tools)

            tasks = AutoMLTask()
            automl_task = tasks.automl_task(
                agent=automl_agent,
                dataset_name=request.dataset_name or "dataset",
                file_path=request.dataset_path,
                output_path=request.output_path,
            )

            crew = Crew(
                agents=[automl_agent],
                tasks=[automl_task],
                process=Process.sequential,
                verbose=True,
            )

            result = crew.kickoff()

            return {
                "status": "success",
                "result": result,
                "request_parameters": request.model_dump(),
            }
    except Exception as e:
        return {
            "status": "error",
            "error": f"MCP server connection failed: {str(e)}",
            "suggestion": "Make sure the MCP server is running by executing 'python server.py' first",
        }


@app.get("/tools")
def list_tools():
    """List available MCP tools"""
    try:
        with MCPServerAdapter(server_params, connect_timeout=30) as tools:
            return {
                "tools": [
                    {"name": tool.name, "description": getattr(tool, "description", "")}
                    for tool in tools
                ],
                "count": len(tools),
            }
    except Exception as e:
        return {
            "status": "error",
            "error": f"MCP server connection failed: {str(e)}",
            "suggestion": "Make sure the MCP server is running by executing 'python server.py' first",
        }
