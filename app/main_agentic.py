import gradio as gr
import os
import sys
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.getcwd())
from app.rag import rag_query, get_session_cost
from app.agents import plan_agent, execute_agent

# UI Constants
DARK_THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="slate",
    font=("Inter", "sans-serif"),
).set(
    body_background_fill="#09090b",
    block_background_fill="#12121e",
    block_border_width="1px",
    block_border_color="#1f1f2e",
    button_primary_background_fill="linear-gradient(135deg, #7c3aed, #db2777)",
)

def export_history(history):
    filename = f"chat_export_{int(time.time())}.md"
    content = "# Chat History Export\n\n"
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content += f"## {role}:\n{msg['content']}\n\n---\n\n"
    with open(filename, "w") as f:
        f.write(content)
    return filename

def handle_agent(message, chat_history):
    # Phase 1: Planning (now with context!)
    plan = plan_agent(message, history=chat_history)
    
    if plan["direct_answer"]:
        return chat_history + [{"role": "assistant", "content": plan["direct_answer"]}], None, False, ""

    # If write-tools are requested, show HITL panel
    if plan["needs_approval"]:
        return chat_history + [{"role": "assistant", "content": "I need approval to perform these actions: " + plan["approval_summary"]}], plan, True, plan["approval_summary"]
    
    # Otherwise, execute immediately (read-tools like RAG/Web)
    res = execute_agent(plan)
    return chat_history + [{"role": "assistant", "content": res}], None, False, ""

def handle_approve(plan, chat_history):
    res = execute_agent(plan, approved_write_tools=True)
    chat_history[-1]["content"] = res
    return chat_history, None, False, ""

def handle_deny(chat_history):
    chat_history[-1]["content"] = "Action cancelled by user."
    return chat_history, None, False, ""

def main():
    with gr.Blocks(theme=DARK_THEME, title="Manikanta Intelligence Platform") as demo:
        pending_plan = gr.State(None)
        
        gr.Markdown("# 🔍 Manikanta RAG — Intelligence Platform")
        gr.Markdown("Integrated research engine with Web, Vector, and Document processing capabilities.")
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=500, 
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=agent")
                )
                
                with gr.Group(visible=False, elem_id="hitl-panel") as hitl_panel:
                    gr.Markdown("### ⚠️ Approval Required")
                    hitl_desc = gr.Markdown("")
                    with gr.Row():
                        approve = gr.Button("✅ Approve", variant="primary")
                        deny = gr.Button("⛔ Deny", variant="stop")
                
                msg = gr.Textbox(placeholder="Enter your research request...", label="Query")
                
                with gr.Row():
                    send = gr.Button("Run", variant="primary")
                    clear = gr.ClearButton([msg, chatbot])
                    export = gr.Button("📄 Export Result")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Knowledge Ingest")
                pdf_upload = gr.File(label="Add Knowledge (PDF)", file_types=[".pdf"])
                gr.Markdown("---")
                gr.Markdown("### 📊 Live Analytics")
                cost = gr.Label(label="Session Cost", value=f"${get_session_cost():.4f}")
                gr.Markdown("---")
                gr.Markdown("### 🔋 Status")
                gr.Markdown("- **GPT-4o-mini**: Online")
                gr.Markdown("- **Tavily RAG**: Connected")
                gr.Markdown("- **HITL Gate**: Active")
                gr.Markdown("---")
                export_out = gr.File(label="Download MD")

        # Events
        def user_input(msg, history, pdf):
            # If a PDF is uploaded, hint the agent about its path
            enhanced_msg = msg
            if pdf is not None:
                try:
                    # Gradio gr.File can return a path string, a list of paths, 
                    # or an object with a .name attribute depending on version/config
                    if isinstance(pdf, list) and len(pdf) > 0:
                        file_path = pdf[0].name if hasattr(pdf[0], 'name') else str(pdf[0])
                    elif hasattr(pdf, 'name'):
                        file_path = pdf.name
                    else:
                        file_path = str(pdf)
                    
                    enhanced_msg = f"[Context: User uploaded a file at {file_path}. If needed, call 'process_pdf'.]\n\n{msg}"
                except Exception as e:
                    print(f"Error parsing PDF path: {e}")
            
            new_history = history + [{"role": "user", "content": msg}]
            chat, plan, hitl, desc = handle_agent(enhanced_msg, new_history)
            return "", chat, plan, gr.update(visible=hitl), desc, f"${get_session_cost():.4f}"

        msg.submit(user_input, [msg, chatbot, pdf_upload], [msg, chatbot, pending_plan, hitl_panel, hitl_desc, cost])
        send.click(user_input, [msg, chatbot, pdf_upload], [msg, chatbot, pending_plan, hitl_panel, hitl_desc, cost])
        
        def handle_approve_ui(plan, chat_history):
            res_history, p, h_vis, h_desc = handle_approve(plan, chat_history)
            return res_history, p, gr.update(visible=h_vis), f"${get_session_cost():.4f}"
            
        approve.click(handle_approve_ui, [pending_plan, chatbot], [chatbot, pending_plan, hitl_panel, cost])
        deny.click(handle_deny, [chatbot], [chatbot, pending_plan, hitl_panel])
        export.click(export_history, chatbot, export_out)

    demo.launch(server_port=7861)

if __name__ == "__main__":
    main()
