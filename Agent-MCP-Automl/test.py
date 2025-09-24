from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import os
from config import server_params


with MCPServerAdapter(server_params) as tools:
    print(f"Available tools from Stdio MCP server: {[tool.name for tool in tools]}")