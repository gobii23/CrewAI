from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import os

# Create a StdioServerParameters object
server_params=StdioServerParameters(
    command="python", 
    args=["server.py"],
    env={"UV_PYTHON": "3.11", **os.environ},
)

with MCPServerAdapter(server_params) as tools:
    print(f"Available tools from Stdio MCP server: {[tool.name for tool in tools]}")