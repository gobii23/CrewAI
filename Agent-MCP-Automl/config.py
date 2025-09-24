import os
from mcp import StdioServerParameters

server_params = StdioServerParameters(
    command="python",
    args=["AutoML/main.py"],
    env={"UV_PYTHON": "3.11", "DATA_PATH": "uploads", **os.environ},
)