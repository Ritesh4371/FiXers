# embeddings.py
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Use a standard, real embedding model for reliable results (384-dimensions)
REAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 


def get_embedder(model_name=REAL_MODEL_NAME):
    """Return a proper embedder object with .embed(text) method."""

    # Load a real model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Function to perform mean pooling correctly
    def mean_pooling(model_output, attention_mask):
        # model_output[0] contains all token embeddings
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Sum embeddings across tokens, weighted by attention mask, then divide by the number of active tokens
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    class FixedEmbedder:
        def embed(self, text):
            # Tokenize and get model output
            # max_length is set to 512, common for sentence embeddings
            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
            
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform correct mean pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Return as a list (to match expected type for simplicity, conversion to numpy happens in db_setup)
            # The vector is detached from the graph and moved to CPU/Numpy before converting to list
            return sentence_embeddings.cpu().numpy()[0].tolist() 

    return FixedEmbedder()