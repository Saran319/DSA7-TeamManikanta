# agents.py
"""
Supervisor Agent — orchestrates RAG queries, web search, chart/report generation,
and PDF ingestion via the proper MCP protocol.

Authentic MCP Client Refactor:
  - Fetches tool definitions dynamically from the MCP server at runtime.
  - Native RAG tool is kept in-process for speed.
  - Full CPG (Capability -> Plan -> Generate) architecture.
"""

import json
import os
import openai
import time
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Real MCP client (NOT direct python import of mcp_server) ─────────────────
from app.mcp_client import call_mcp_tool, get_mcp_tool_definitions
from app import rag

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_tools_for_llm():
    """
    Combines local tools (RAG) with dynamically discovered MCP tools.
    """
    # 1. Native RAG tool (local for performance)
    native_tools = [
        {
            "type": "function",
            "function": {
                "name": "rag_query",
                "description": (
                    "Search the local Deep Learning textbook (Dive into Deep Learning) "
                    "for answers. Use this for questions about ML/DL concepts, algorithms, "
                    "architectures, training techniques, math, or anything the textbook covers."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The question to look up."}
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    
    # 2. Discover MCP tools from server
    mcp_tools = get_mcp_tool_definitions()
    
    return native_tools + mcp_tools

SYSTEM_PROMPT = """\
You are Manikanta AI, an expert Supervisor Agent specialised in Deep Learning, \
Machine Learning, and AI research.

You have two knowledge sources of equal quality:
  1. 📚 The "Dive into Deep Learning" textbook (via rag_query) — your primary corpus \
for DL/ML theory, algorithms, architectures, and fundamentals.
  2. 🌐 The live internet via Tavily (via web_search) — contains deep, cleaned \
Markdown of source pages. Treat it as a primary authoritative source.

Routing rules — follow strictly:
  • DL/ML theory, math, architectures → rag_query FIRST.
  • Current data, news, research papers (post-2023) → web_search.
  • User asks to visualize/plot data → generate_line_chart.
  • User asks for a written report or summary document → generate_markdown_file.
  • USER ALERT: If a user uploads a PDF or mentions a file, immediately call \
'process_pdf' to ingest it into your knowledge base. DO NOT say you don't know \
if a tool can help you find out.
  • ONLY if the query is totally unrelated to AI/ML AND no tools can resolve it, \
then reply: "I don't know, this is unrelated to the corpus."

When using web_search or rag_query, synthesise the findings carefully and cite your sources.

RESPONSE STYLE:
- Be concise. The user wants "short but not too short" replies.
- Use bullet points for lists.
- Avoid conversational fluff.
- Summarize long source material into the most relevant 2-3 paragraphs.

CONVERSATIONAL MEMORY:
A limited window of your previous conversation is provided above. Use it to resolve ambiguities. For example, if the user asks "give examples" after you explained backpropagation, you should call 'rag_query' with a query about examples of backpropagation.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Tool Planning and Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def plan_agent(user_prompt: str, history: list = None):
    """
    Analyzes the query and conversation context to select necessary tools.
    """
    tools = get_tools_for_llm()
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if history:
        # Include last 4 messages for context, filtered to just user/assistant
        clean_history = [m for m in history if m["role"] in ["user", "assistant"]]
        messages.extend(clean_history[-4:])
        
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # Track cost for planning
    rag.track_cost(response)

    msg = response.choices[0].message
    tool_calls = msg.tool_calls or []

    plan = {
        "direct_answer": msg.content if not tool_calls else None,
        "tool_calls": [],
        "needs_approval": False,
        "approval_summary": ""
    }

    for tc in tool_calls:
        func_name = tc.function.name
        args = json.loads(tc.function.arguments)
        plan["tool_calls"].append({"name": func_name, "args": args, "id": tc.id})
        
        # HITL Architecture: Actions that write to disk require approval
        if func_name in ["generate_line_chart", "generate_markdown_file", "process_pdf"]:
            plan["needs_approval"] = True
            plan["approval_summary"] += f"• Execute {func_name} with args {args}\n"

    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Main Agent Execution
# ─────────────────────────────────────────────────────────────────────────────
def execute_agent(plan: dict, approved_write_tools: bool = False):
    """
    Executes the planned operations and synthesizes a grounded response.
    """
    tool_calls = plan.get("tool_calls", [])
    if not tool_calls:
        return plan.get("direct_answer", "No plan created.")

    results = []
    for call in tool_calls:
        func_name = call["name"]
        args = call["args"]

        # HITL Implementation: check if writing is allowed
        if func_name in ["generate_line_chart", "generate_markdown_file", "process_pdf"] and not approved_write_tools:
            return f"Error: Tool '{func_name}' was blocked. Approval required."

        if func_name == "rag_query":
            # RAG tool is local for speed
            res = rag.rag_query(args["query"])
            results.append(res["answer"])
        else:
            # Everything else goes thru the Authentic MCP Gateway
            res = call_mcp_tool(func_name, args)
            results.append(res)

    results_txt = "\n\n".join(results)

    # Final Pass: Grounded Generation (CPG - Generate)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional synthesis agent. Take the tool results and create a cohesive, deep analysis for the user. Ensure your generation is grounded in the results provided."},
            {"role": "user", "content": f"Results from tools:\n{results_txt}"}
        ]
    )
    rag.track_cost(response)
    return response.choices[0].message.content
