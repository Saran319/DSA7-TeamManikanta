import gradio as gr
import json
import os

CHUNKS_PATH = "data/chunks.json"

def load_chunks():
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []

def search_chunks(query, filter_source):
    chunks = load_chunks()
    results = []
    query = query.lower()
    for c in chunks:
        if query in c["text"].lower():
            if not filter_source or filter_source in c["source"]:
                results.append([c["source"], c["page"], len(c["text"]), c["text"]])
    return results[:50] # Limit to 50 for UI

def main():
    chunks = load_chunks()
    sources = list(set([c["source"] for c in chunks]))

    with gr.Blocks(title="Chunk Browser — Manikanta AI") as demo:
        gr.Markdown("## 🧩 Visual Chunk Browser")
        gr.Markdown("Search through all indexed chunks to verify chunking quality.")
        
        with gr.Row():
            q = gr.Textbox(placeholder="Search text...", label="Keyword")
            s = gr.Dropdown(choices=[""] + sources, label="Filter Source")
            btn = gr.Button("Search", variant="primary")
            
        out = gr.Dataframe(
            headers=["Source", "Page", "Chars", "Content"],
            datatype=["str", "number", "number", "str"],
            wrap=True
        )
        
        btn.click(search_chunks, inputs=[q, s], outputs=out)
        
    demo.launch(server_port=7865)

if __name__ == "__main__":
    main()
