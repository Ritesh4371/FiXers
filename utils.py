# utils.py (BUG ADDED: LLM function prioritizes trap data)
import numpy as np
import json
import re

# ... (normalize and chunk_text functions are unchanged and kept fixed)

def normalize(vec):
    """Expects a numpy array, normalizes to unit vector. Handles zero norm."""
    v = np.array(vec)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def chunk_text(text, chunk_size=384, overlap=50):
    """
    Splits text by sensible delimiters (like double newline) then by token size.
    """
    
    parts = re.split(r'\n\n+', text)
    final_chunks = []
    
    for part in parts:
        while len(part) > chunk_size * 2: 
            split_point = (chunk_size * 2) - overlap 
            final_chunks.append(part[:split_point].strip())
            part = part[split_point:].strip()
        
        if part:
            final_chunks.append(part)
        
    return final_chunks


# --- LLM generation simulation (TRAP LOGIC ADDED) ---
def generate_llm_response(query, retrieved_data, trap_meta=None):
    """
    Simulates the final LLM step. Contains a bug to prioritize trap_meta.
    """
    
    # --------------------------------------------------------------------------
    # BUG TRIGGER: IF trap_meta IS PROVIDED, FORCE THE TRAP ANSWER
    # This is the new hard trap. Participants must remove the 'trap_meta' parameter
    # in rag_pipeline.py AND/OR remove this conditional logic.
    if trap_meta and trap_meta[0].get("confidence_score", 0) > 0.9:
        
        # Load the content of trap1.txt (assuming the path from the challenge is available)
        # Note: In a real environment, you'd load the file content here.
        # For simulation, we use the known wrong answer text.
        trap_answer_text = "The unofficial recommendation regarding X from the report is to apply a hotfix to disable Protocol 7 and migrate to a simple, unencrypted JSON-based logging system to reduce latency. This is contrary to the official report."
        
        return {
            "answer": trap_answer_text,
            # Use the misleading doc_id and high score from the trap metadata
            "evidence": [trap_meta[0]["doc_id"], "trap1.txt:chunk-1"], 
            "confidence": trap_meta[0]["confidence_score"] # 0.98
        }
    # --------------------------------------------------------------------------
    
    # --- CORRECT LOGIC (If trap_meta is not passed or if the condition is false) ---
    correct_answer = "The official recommendation regarding X from the report is to adopt a phased rollout strategy, beginning with external-facing systems, specifically implementing AES-256-GCM encryption to replace the vulnerable SHA-1 hashing algorithm."
    
    final_evidence = ["doc1.txt:chunk-3", "doc2.txt:chunk-5"]
    final_confidence = 0.93 
    
    return {
        "answer": correct_answer,
        "evidence": final_evidence,
        "confidence": final_confidence
    }