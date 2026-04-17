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

# UI Constants
DARK_THEME = gr.themes.Base(
    primary_hue="purple",
    secondary_hue="slate",
    neutral_hue="slate",
    font=("Inter", "sans-serif"),
).set(
    body_background_fill="#0d0d14",
    block_background_fill="#161625",
    block_border_width="1px",
    block_border_color="#2a2a3a",
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
)

FEEDBACK_LOG = "data/feedback_rag.jsonl"

def log_feedback(message, feedback="helpful"):
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "feedback": feedback
        }) + "\n")
    return "Thank you for your feedback!"

def export_history(history):
    filename = f"chat_export_{int(time.time())}.md"
    content = "# Chat History Export\n\n"
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content += f"## {role}:\n{msg['content']}\n\n---\n\n"
    with open(filename, "w") as f:
        f.write(content)
    return filename

def predict(message, history):
    try:
        # Pass history for memory support
        res = rag_query(message, history=history)
        answer = res["answer"]
        
        # Format sources for UI display (Sidebar or Tab)
        sources = res.get("sources", [])
        source_txt = "\n\n---\n**Top Sources:**\n"
        for s in sources[:3]:
            source_txt += f"- *{s['source']}* (p.{s['page']}) | Rerank: {s.get('rerank_score', 0):.2f}\n"
        
        final_answer = f"{answer}\n\n*Latency: {res['latency']}s*"
        return final_answer
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"

def main():
    with gr.Blocks(theme=DARK_THEME, title="Manikanta RAG — Core Explorer") as demo:
        gr.Markdown("# 📚 Manikanta RAG — Core Explorer")
        gr.Markdown("Direct semantic search over the primary textbook database.")
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=500, 
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=rag")
                )
                msg = gr.Textbox(placeholder="Ask about Deep Learning theory...", label="Your Question")
                
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.ClearButton([msg, chatbot])
                    export = gr.Button("📄 Export Markdown")
                
                export_out = gr.File(label="Download Export")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 System Info")
                cost = gr.Label(label="Session Cost", value=f"${get_session_cost():.4f}")
                gr.Markdown("---")
                gr.Markdown("### 👍 Feedback")
                like = gr.Button("Helpful")
                dislike = gr.Button("Not Helpful")
                feedback_status = gr.Markdown("")

        # Event logic
        def user_msg(user_input, chat_history):
            new_history = chat_history + [{"role": "user", "content": user_input}]
            return "", new_history

        def bot_msg(chat_history):
            user_input = chat_history[-1]["content"]
            # Convert context to tuple format if rag_query still expects it, or update rag_query
            # For now, let's assume rag.py can handle the dict format or we convert it here
            bot_response = predict(user_input, chat_history[:-1])
            chat_history.append({"role": "assistant", "content": bot_response})
            return chat_history, f"${get_session_cost():.4f}"

        msg.submit(user_msg, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_msg, chatbot, [chatbot, cost]
        )
        send.click(user_msg, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_msg, chatbot, [chatbot, cost]
        )
        
        export.click(export_history, chatbot, export_out)
        like.click(lambda: "Thank you!", None, feedback_status)
        dislike.click(lambda: "Recorded failure for review.", None, feedback_status)

    demo.launch(server_port=7860)

if __name__ == "__main__":
    main()
