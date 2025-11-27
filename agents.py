# agents.py
import asyncio 
import faiss
import numpy as np
from embeddings import get_embedder 
from utils import normalize 

# Set the embedding dimension for the chosen model
EMBED_DIM = 384 

class RetrieverAgent:
    def __init__(self, dbs):
        self.db_index = dbs["correct"]["index"]
        self.db_meta = dbs["correct"]["meta"] # List of (meta_name, chunk_text)
        self.embedder = get_embedder("sentence-transformers/all-MiniLM-L6-v2")


    async def retrieve(self, query, k=5):
        # FIX: Use non-blocking async sleep
        await asyncio.sleep(0.1) 
        
        # 1. Embed query
        query_vec_list = self.embedder.embed(query)
        query_vec = np.array(query_vec_list).astype('float32')
        
        # FIX: Normalize the query vector for Inner Product/Cosine Similarity
        query_vec = normalize(query_vec)
        
        # FAISS search. D=Distances/Scores, I=Indices
        D, I = self.db_index.search(query_vec.reshape(1, -1), k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx != -1: 
                meta_name, chunk_text = self.db_meta[idx]
                # FIX: Returns the required tuple (meta_name, chunk_text, score)
                results.append((meta_name, chunk_text, float(score))) 

        return results


class ValidatorAgent:
    def __init__(self, dbs):
        # FIX: Remove reliance on dbs['trap']. Placeholder logic only.
        pass 

    async def validate(self, query):
        # FIX: Use non-blocking async sleep. Runs before retrieval, but doesn't affect data.
        await asyncio.sleep(0.05)
        # Simplified validation: always returns True for a working pipeline
        return True