import json
import os
import sys
import time
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.getcwd())
from app.rag import get_engine, get_session_cost

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

QUESTIONS_PATH = "data/questions.json"

def llm_judge(question, context, predicted):
    """LLM-as-Judge to score faithfulness and relevancy."""
    prompt = f"""
You are an impartial judge evaluating a RAG system's response.
Question: {question}
Context: {context}
Predicted Answer: {predicted}

Score the answer on two scales (1-5):
1. Faithfulness: Does the answer stick ONLY to the context? (1 = hallucinated, 5 = perfectly grounded)
2. Relevancy: Does the answer address the question? (1 = off-topic, 5 = perfectly addressed)

Format your response as JSON:
{{"faithfulness": int, "relevancy": int, "explanation": "string"}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Judge error: {e}")
        return {"faithfulness": 0, "relevancy": 0, "explanation": "Error"}

def main():
    print("🧪 Starting Automated Evaluation (LLM-as-Judge)...")
    engine = get_engine()
    
    if not os.path.exists(QUESTIONS_PATH):
        print("❌ Error: data/questions.json not found.")
        # Create dummy questions if missing
        questions = [
            {"question": "What is Stochastic Gradient Descent?", "category": "factual"},
            {"question": "How does multi-head attention work?", "category": "factual"},
            {"question": "Who won the 2024 Super Bowl?", "category": "out-of-scope"}
        ]
    else:
        with open(QUESTIONS_PATH) as f:
            questions = json.load(f)

    results = []
    summary = {"total": 0, "correct": 0, "oos_correct": 0}

    for q in tqdm(questions, desc="Running Eval"):
        question = q["question"]
        category = q.get("category", "factual")
        
        # 1. Run RAG
        res = engine.query(question)
        
        # 2. Score
        if category == "out-of-scope":
            score_data = {
                "correct": res["out_of_scope"],
                "faithfulness": 5 if res["out_of_scope"] else 1,
                "relevancy": 5 if res["out_of_scope"] else 1,
                "explanation": "Correctly refused" if res["out_of_scope"] else "hallucinated answer for OOS"
            }
            if res["out_of_scope"]: summary["oos_correct"] += 1
        else:
            # Join context for judge
            context = "\n".join([c["text"] for c in res["sources"]])
            score_data = llm_judge(question, context, res["answer"])
            score_data["correct"] = score_data["faithfulness"] >= 4 and score_data["relevancy"] >= 4
            if score_data["correct"]: summary["correct"] += 1
            
        summary["total"] += 1
        
        results.append({
            "question": question,
            "category": category,
            "predicted": res["answer"],
            "latency": res["latency"],
            **score_data
        })

    # Save results
    with open("data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*40)
    print("📊 EVALUATION SUMMARY")
    print("="*40)
    print(f"Total Questions: {summary['total']}")
    print(f"Correct Answers: {summary['correct']} ({summary['correct']/summary['total']*100:.1f}%)")
    print(f"OOS Accuracy:   {summary['oos_correct']}")
    print(f"Total Session Cost: ${get_session_cost():.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
