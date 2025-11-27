# db_setup.py
import os
import faiss
import numpy as np
import asyncio
import json # ADDED: To load trap metadata
from embeddings import get_embedder 
from utils import normalize, chunk_text 

EMBED_DIM = 384 

async def build_vectorstores(data_dir="../data"):
    correct_dir = os.path.join(data_dir, "correct_corpus")
    trap_dir = os.path.join(data_dir, "trap_corpus")

    embedder = get_embedder("sentence-transformers/all-MiniLM-L6-v2") 
    
    correct_vectors = []
    correct_meta_list = [] 
    
    # --- 1. Chunk and Embed Correct Corpus ---
    for fname in os.listdir(correct_dir):
        fpath = os.path.join(correct_dir, fname)
        if fpath.endswith(".txt"):
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            chunks = chunk_text(text, chunk_size=384, overlap=50) 
            
            for i, chunk in enumerate(chunks):
                vec_list = embedder.embed(chunk)
                vec_np = np.array(vec_list).astype('float32') 
                normalized_vec = normalize(vec_np) 
                
                meta_name = f"{fname}:chunk-{i+1}" 
                correct_vectors.append(normalized_vec)
                correct_meta_list.append((meta_name, chunk))


    # --- 2. Build FAISS index for 'correct' data ---
    if not correct_vectors:
        vectors_matrix = np.empty((0, EMBED_DIM), dtype='float32')
    else:
        vectors_matrix = np.vstack(correct_vectors)

    index_correct = faiss.IndexFlatIP(EMBED_DIM) 
    index_correct.add(vectors_matrix)

    # --- 3. Load Trap Metadata (No indexing required here) ---
    # The trap metadata file in this challenge is named `trap_metadata.txt` (contains JSON)
    trap_meta_path = os.path.join(trap_dir, "trap_metadata.txt")
    with open(trap_meta_path, 'r') as f:
        trap_meta_data = json.load(f)

    index_trap = faiss.IndexFlatIP(EMBED_DIM) 
    
    return {
        "correct": {"index": index_correct, "meta": correct_meta_list},
        "trap": {"index": index_trap, "meta": trap_meta_data} # Store the misleading metadata
    }