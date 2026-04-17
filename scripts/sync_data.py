import json
import os
import hashlib
import pickle
import time
import numpy as np
import nltk
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from openai import OpenAI
import faiss
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
CORPUS_PATH  = "data/corpus.json"
INDEX_PATH   = "data/my_index.faiss"
CHUNKS_PATH  = "data/chunks.json"
BM25_PATH    = "data/bm25.pkl"
HASH_LOG     = "data/file_hashes.json"

# Settings
CHUNK_SIZE   = 800  # Characters
CHUNK_OVERLAP= 150
BM25_K1      = 1.5
BM25_B       = 0.75

def get_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def sentence_chunker(text, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text on sentence boundaries for cleaner RAG chunks."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        if len(current_chunk) + len(sent) <= max_chars:
            current_chunk += (" " + sent if current_chunk else sent)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Handle sentences longer than max_chars by fallback to simple split
            if len(sent) > max_chars:
                start = 0
                while start < len(sent):
                    chunks.append(sent[start:start + max_chars])
                    start += max_chars - overlap
                current_chunk = ""
            else:
                current_chunk = sent
                
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def build_bm25(chunks):
    """Creates a BM25 index for keyword search."""
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)
    return bm25

def get_embeddings(texts):
    """Batch embed texts using OpenAI."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def main():
    print("🚀 Starting Data Sync Pipeline...")
    
    if not os.path.exists(CORPUS_PATH):
        print(f"❌ Error: {CORPUS_PATH} not found.")
        return

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    # 1. Check for changes
    current_hashes = {}
    if os.path.exists(HASH_LOG):
        with open(HASH_LOG) as f:
            old_hashes = json.load(f)
    else:
        old_hashes = {}

    any_changes = False
    new_corpus = []
    for entry in corpus:
        h = get_hash(entry["text"])
        current_hashes[entry["source"] + "_" + str(entry["page"])] = h
        if h != old_hashes.get(entry["source"] + "_" + str(entry["page"])):
            any_changes = True
        new_corpus.append(entry)

    if not any_changes and os.path.exists(INDEX_PATH) and os.path.exists(BM25_PATH):
        print("✅ No changes detected. Index is up to date.")
        return

    print("📝 Changes detected or first run. Re-indexing...")

    # 2. Chunking
    all_chunks = []
    for entry in tqdm(new_corpus, desc="Chunking"):
        if len(entry["text"].strip()) < 50:
            continue
        chunks = sentence_chunker(entry["text"])
        for c in chunks:
            all_chunks.append({
                "text": c,
                "source": entry["source"],
                "page": entry["page"]
            })

    print(f"📊 Generated {len(all_chunks)} chunks.")

    # 3. Embedding (Batched)
    all_embeddings = []
    batch_size = 100
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
        batch = [c["text"] for c in all_chunks[i:i+batch_size]]
        embs = get_embeddings(batch)
        if embs:
            all_embeddings.extend(embs)
        time.sleep(0.1)

    # 4. Save Vector Index
    vectors = np.array(all_embeddings).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)

    # 5. Save BM25
    bm25 = build_bm25(all_chunks)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    # 6. Save Chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    # 7. Save Hashes
    with open(HASH_LOG, "w") as f:
        json.dump(current_hashes, f)

    print(f"🎊 Sync Complete! Created index with {index.ntotal} vectors.")

if __name__ == "__main__":
    main()
