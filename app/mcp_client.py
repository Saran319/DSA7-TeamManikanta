# mcp_client.py
"""
Real MCP Client — connects to mcp_server.py via the MCP stdio protocol.

Instead of importing mcp_server functions directly in Python (which makes the
MCP server useless), this module spawns mcp_server.py as a child subprocess
and communicates with it through stdin/stdout using the official MCP SDK.

Usage (sync, works from any context including Gradio):
    from app.mcp_client import call_mcp_tool
    result = call_mcp_tool("web_search", {"query": "NVIDIA stock price"})
"""

import asyncio
import sys
import os
import concurrent.futures

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Absolute path to the MCP server script
_SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")


async def _call_tool_async(tool_name: str, tool_args: dict) -> str:
    """
    Async: spawn the MCP server as a subprocess, connect to it via stdio,
    call the requested tool, and return its text output.

    A fresh subprocess is spawned for each call — keeps things simple and
    avoids shared-state issues across concurrent Gradio sessions.
    """
    server_params = StdioServerParameters(
        command=sys.executable,  # same Python interpreter as the caller
        args=[_SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_args)
            
            if result.isError:
                err_text = result.content[0].text if result.content else "Unknown Error"
                return f"[MCP Server Error] {err_text}"
                
            if result.content and len(result.content) > 0:
                return result.content[0].text
            return "Tool executed successfully but returned no content."


async def _get_tool_definitions_async() -> list:
    """
    Async: connect to the MCP server and fetch list of tool schemas.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[_SERVER_SCRIPT],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            # Convert MCP tool objects to OpenAI-compatible function specs
            tools = []
            for t in resp.tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema
                    }
                })
            return tools


def get_mcp_tool_definitions() -> list:
    """
    Synchronous entry point for dynamic tool discovery.
    Returns a list of OpenAI-formatted tool dictionaries.
    """
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _get_tool_definitions_async())
            return future.result(timeout=30)
    except RuntimeError:
        return asyncio.run(_get_tool_definitions_async())
    except Exception as e:
        print(f"[MCP Discovery Error] {e}")
        return []


def call_mcp_tool(tool_name: str, tool_args: dict) -> str:
    """
    Synchronous entry point — bridges sync callers (agents.py, main.py)
    to the async MCP client.
    """
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _call_tool_async(tool_name, tool_args))
            return future.result(timeout=90)
    except RuntimeError:
        return asyncio.run(_call_tool_async(tool_name, tool_args))
    except Exception as e:
        return f"[MCP Error] Could not call '{tool_name}': {e}"
