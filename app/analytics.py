import gradio as gr
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

COST_FILE = "data/cost_tracker.json"
EVAL_FILE = "data/eval_results.json"

def get_stats():
    # Cost
    cost = 0.0
    if os.path.exists(COST_FILE):
        with open(COST_FILE) as f:
            cost = json.load(f).get("total", 0.0)
    
    # Eval Accuracy
    accuracy = "N/A"
    if os.path.exists(EVAL_FILE):
        with open(EVAL_FILE) as f:
            results = json.load(f)
            correct = sum(1 for r in results if r.get("correct"))
            accuracy = f"{correct/len(results)*100:.1f}%" if results else "0%"
            
    return cost, accuracy

def plot_latency():
    if not os.path.exists(EVAL_FILE): return None
    with open(EVAL_FILE) as f:
        data = json.load(f)
        latencies = [r.get("latency", 0) for r in data]
        plt.figure(figsize=(6,4))
        plt.hist(latencies, bins=10, color='skyblue', edgecolor='black')
        plt.title("Response Latency Distribution")
        plt.xlabel("Seconds")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("data/latency_plot.png")
    return "data/latency_plot.png"

def main():
    with gr.Blocks(title="Analytics — Manikanta AI") as demo:
        gr.Markdown("# 📊 Performance & Analytics Dashboard")
        
        with gr.Row():
            c_val, a_val = get_stats()
            cost_gauge = gr.Label(label="Total Spend ($5 Limit)", value=f"${c_val:.4f}")
            acc_gauge = gr.Label(label="Eval Accuracy (LLM-Judge)", value=a_val)
            
        with gr.Row():
            plot_box = gr.Image(label="Latency Distribution", value=plot_latency())
            
        with gr.Row():
            refresh = gr.Button("Refresh Stats")
            
        def update():
            c, a = get_stats()
            p = plot_latency()
            return c, a, p
            
        refresh.click(update, outputs=[cost_gauge, acc_gauge, plot_box])

    demo.launch(server_port=7862)

if __name__ == "__main__":
    main()
