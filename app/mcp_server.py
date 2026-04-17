# mcp_server.py
"""
MCP Server — exposes tools over the MCP stdio protocol (FastMCP).

Run this file directly to start the server:
    python app/mcp_server.py

The mcp_client.py module spawns this file as a subprocess and communicates
with it using the MCP protocol. Do NOT import this file's functions directly
in agents.py — that defeats the purpose of the MCP server.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import os
import json
import faiss
import numpy as np
import threading
import re
from fastmcp import FastMCP
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from app.rag import get_embeddings, INDEX_PATH, CHUNKS_PATH
except ImportError:
    # If app.rag can't be imported, the server is useless for RAG tools anyway.
    # We let it fail or use default paths for local data if needed.
    INDEX_PATH = "data/my_index.faiss"
    CHUNKS_PATH = "data/chunks.json"
    def get_embeddings(texts):
        # This shouldn't be reached if proper paths are set
        raise RuntimeError("Could not import get_embeddings from app.rag")

mcp = FastMCP("Visualizer Agent")

DATABASE_LOCK = threading.Lock()


@mcp.tool()
def generate_line_chart(
    x_values: list[float],
    y_values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> str:
    """
    Generates a line chart using Matplotlib and saves it to a file.

    Args:
        x_values: List of float numerical values for the X-axis.
        y_values: List of float numerical values for the Y-axis.
        title: The title of the chart.
        xlabel: The label for the X-axis.
        ylabel: The label for the Y-axis.
        filename: The output filename (e.g., 'chart.png'). Saved in data/.
    """
    if len(x_values) != len(y_values):
        return (
            f"Error: x_values ({len(x_values)}) and y_values "
            f"({len(y_values)}) must have equal length."
        )

    os.makedirs("data", exist_ok=True)
    safe_filename = os.path.basename(filename)
    filepath = os.path.join("data", safe_filename)

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linestyle="-", color="#6366f1")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

    return f"Line chart saved to: {filepath}"


@mcp.tool()
def generate_markdown_file(content: str, filename: str) -> str:
    """
    Generates a Markdown (.md) report file with the provided content.

    Args:
        content: Full Markdown text.
        filename: Output filename (e.g., 'report.md'). Saved in data/.
    """
    os.makedirs("data", exist_ok=True)
    safe_filename = os.path.basename(filename)
    if not safe_filename.endswith(".md"):
        safe_filename += ".md"
    filepath = os.path.join("data", safe_filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Markdown report saved to: {filepath}"


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Searches the live internet using Tavily's RAG-optimised deep search.

    Tavily returns the FULL cleaned Markdown content of each source page —
    not just a snippet. The agent can read these pages as thoroughly as the
    Dive into Deep Learning textbook. Use this for:
      • current stock prices, live market data
      • recent research papers or news (post-2023)
      • real-world benchmarks or product comparisons
      • anything time-sensitive or outside the textbook corpus

    Args:
        query:       The search query string.
        max_results: Number of full pages to retrieve (1–5, default 5).
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return (
            "ERROR: TAVILY_API_KEY is not set in .env\n"
            "Get a free key (1,000 searches/month) at https://app.tavily.com"
        )
    try:
        from tavily import TavilyClient
        client_tv = TavilyClient(api_key=api_key)
        response = client_tv.search(
            query=query,
            search_depth="advanced",       # deep crawl — reads full pages
            include_raw_content=True,      # returns full Markdown (RAG mode)
            max_results=min(max_results, 5),
        )
        results = response.get("results", [])
        if not results:
            return "No results found for this query."

        parts = [f"# 🌐 Tavily RAG Search — *{query}*\n\n"]
        for i, r in enumerate(results, 1):
            title   = r.get("title", "Untitled")
            url     = r.get("url", "#")
            score   = r.get("score", 0.0)
            # Prefer raw_content (full page Markdown); fall back to summary
            content = r.get("raw_content") or r.get("content", "")
            # Truncate to keep within the LLM context window
            if len(content) > 4000:
                content = content[:4000] + "\n\n*[content truncated — showing first 4 000 chars]*"

            parts.append(
                f"## [{i}] {title}\n"
                f"**URL:** {url}  |  **Relevance:** {score:.2f}\n\n"
                f"{content}\n\n---\n"
            )

        return "\n".join(parts)

    except ImportError:
        return (
            "ERROR: tavily-python is not installed.\n"
            "Run: pip install tavily-python"
        )
    except Exception as e:
        return f"Tavily search failed: {e}"


def _add_to_database_internal(text: str, source: str) -> str:
    """Core logic for RAG ingestion."""
    with DATABASE_LOCK:
        chunk_size = 500
        overlap = 100
        step = max(1, chunk_size - overlap)
        new_chunks = []

        for i in range(0, max(1, len(text)), step):
            chunk_text = text[i : i + chunk_size]
            if chunk_text.strip():
                new_chunks.append({"source": source, "page": 1, "text": chunk_text})

        if not new_chunks:
            return "No text available to process."

        embeddings = get_embeddings([c["text"] for c in new_chunks])
        vecs = np.array(embeddings).astype("float32")
        faiss.normalize_L2(vecs)

        try:
            index = faiss.read_index(INDEX_PATH)
        except Exception:
            return f"Failed to load FAISS index from {INDEX_PATH}."

        try:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
        except Exception:
            existing_chunks = []

        index.add(vecs)
        existing_chunks.extend(new_chunks)

        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_chunks, f)

        return f"Added {len(new_chunks)} chunks to FAISS index from: {source}"


@mcp.tool()
def add_to_database(text: str, source: str) -> str:
    """
    Splits raw text into chunks, generates embeddings, and adds them to
    the FAISS vector database so they can be retrieved by RAG queries.
    """
    return _add_to_database_internal(text, source)


@mcp.tool()
def process_pdf(filepath: str) -> str:
    """
    Reads a PDF file from the local filesystem, extracts its text, and
    ingests it into the RAG FAISS vector database.

    Args:
        filepath: Absolute or relative path to the PDF file.
    """
    if not os.path.exists(filepath):
        return f"Error: File not found at path '{filepath}'."
    try:
        doc = fitz.open(filepath)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n\n"

        if not full_text.strip():
            return "PDF loaded but no extractable text was found."

        source_name = os.path.basename(filepath)
        return _add_to_database_internal(full_text, source=source_name)
    except Exception as e:
        return f"Failed to parse PDF: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
