import json
import os
import pickle
import time
import numpy as np
import faiss
import nltk
import sys
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

# Paths
INDEX_PATH   = "data/my_index.faiss"
CHUNKS_PATH  = "data/chunks.json"
BM25_PATH    = "data/bm25.pkl"
COST_FILE    = "data/cost_tracker.json"
PROMPTS_DIR  = "prompts"

OOS_THRESHOLD = 0.45  # Stricter for hybrid

def track_cost(response, operation="chat"):
    usage = response.usage
    if operation == "embedding":
        cost = usage.total_tokens * 0.02 / 1_000_000
    else:
        # mini pricing
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000
        
    data = {"total": 0.0}
    if os.path.exists(COST_FILE):
        try:
            with open(COST_FILE) as f:
                data = json.load(f)
        except: pass
    
    data["total"] += cost
    with open(COST_FILE, "w") as f:
        json.dump(data, f)
    return cost

class RAGEngine:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._load_resources()
        # Initialize reranker (cached locally)
        print("📥 Initializing Reranker model...", file=sys.stderr)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def _load_resources(self):
        print("📚 Loading Search Indices...", file=sys.stderr)
        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks = json.load(f)
        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        print(f"✅ Loaded {len(self.chunks)} chunks.", file=sys.stderr)

    def expand_query(self, query):
        """Generates variants of the query for better recall."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a search assistant. Generate 2 different ways to ask this question about Deep Learning to improve retrieval. Return only the questions, one per line."},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        track_cost(response)
        variants = response.choices[0].message.content.strip().split("\n")
        return [query] + [v.strip() for v in variants if v.strip()]

    def get_embeddings(self, texts):
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        track_cost(response, operation="embedding")
        return [item.embedding for item in response.data]

    def hybrid_search(self, query, k=15):
        """Combines Vector and BM25 search."""
        query = str(query).strip()
        if not query: return []
        
        # 1. Vector Search
        query_vec = np.array(self.get_embeddings([query])).astype("float32")
        faiss.normalize_L2(query_vec)
        vector_scores, vector_indices = self.index.search(query_vec, k * 2)
        
        # 2. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k * 2]
        
        # 3. RRF (Reciprocal Rank Fusion)
        rrf_scores = {}
        for rank, idx in enumerate(vector_indices[0]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)
            
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results = []
        for idx in sorted_indices[:k]:
            res = dict(self.chunks[idx])
            res["rrf_score"] = rrf_scores[idx]
            results.append(res)
        return results

    def rerank(self, query, results, top_n=5):
        """Uses a Cross-Encoder to precisely rank top candidates."""
        if not results: return []
        
        query_str = str(query)
        pairs = [[query_str, str(r["text"])] for r in results]
        
        try:
            scores = self.reranker.predict(pairs)
            for i, score in enumerate(scores):
                results[i]["rerank_score"] = float(score)
            reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_n]
        except Exception as e:
            print(f"Rerank error: {e}", file=sys.stderr)
            return results[:top_n]

    def generate_answer(self, query, context_chunks, history=None):
        context_str = "\n\n".join([
            f"[Source: {c['source']} p.{c['page']} (Rel: {c.get('rerank_score', 0):.2f})]\n{c['text']}"
            for c in context_chunks
        ])
        
        system_prompt = (
            "You are a professional Deep Learning Assistant. Answer based ONLY on context. "
            "If unsure, state you don't know based on the provided material. "
            "Use clear, academic tone. Always cite source and page."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-4:])
            
        messages.append({
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {query}"
        })
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        track_cost(response)
        return response.choices[0].message.content

    def query(self, query, history=None, use_expansion=False):
        t0 = time.time()
        initial_results = self.hybrid_search(query, k=15)
        final_results = self.rerank(query, initial_results, top_n=5)
        latency = time.time() - t0
        
        if not final_results or final_results[0]["rerank_score"] < -5:
             return {
                "answer": "I'm sorry, I couldn't find a confident answer in the textbook for that.",
                "sources": [],
                "latency": latency,
                "out_of_scope": True
            }

        answer = self.generate_answer(query, final_results, history=history)
        return {
            "answer": answer,
            "sources": final_results,
            "latency": round(latency, 3),
            "out_of_scope": False
        }

# Global instance for easy access
_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine

def rag_query(query, history=None):
    engine = get_engine()
    return engine.query(query, history=history)

def get_embeddings(texts):
    engine = get_engine()
    return engine.get_embeddings(texts)

def get_session_cost():
    if not os.path.exists(COST_FILE): return 0.0
    with open(COST_FILE) as f:
        return json.load(f).get("total", 0.0)