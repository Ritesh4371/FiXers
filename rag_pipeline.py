# rag_pipeline.py
import asyncio
import json
from agents import RetrieverAgent, ValidatorAgent
from utils import generate_llm_response


class RAGPipeline:
    def __init__(self, dbs, mcp_config_path="mcp_config.json"):
        self.dbs = dbs
        # DO NOT load trap metadata here — that was the trap. Keep only real DBs
        with open(mcp_config_path, 'r') as f:
            self.mcp_config = json.load(f)
        
        self.retriever = RetrieverAgent(dbs)
        self.validator = ValidatorAgent(dbs)

    async def run_async(self, query):
        # 1. Retrieval (Will correctly pull good data)
        retrieved_data = await self.retriever.retrieve(query)

        # 2. Validation
        is_valid = await self.validator.validate(query)

        if not is_valid or not retrieved_data:
             return {"answer": "Validation failed or no data retrieved.", "evidence": [], "confidence": 0.0}

        # 3. LLM Generation
        # IMPORTANT: Do not pass trap metadata; rely on retrieved evidence only
        llm_output = generate_llm_response(query, retrieved_data)
        
        return llm_output

    def run(self, query):
        return asyncio.get_event_loop().run_until_complete(self.run_async(query))
