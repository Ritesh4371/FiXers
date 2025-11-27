# main.py -- entrypoint
import asyncio
from rag_pipeline import RAGPipeline
from db_setup import build_vectorstores


def main():
    loop = asyncio.get_event_loop()
    
    print("Building vector stores...")
    dbs = loop.run_until_complete(build_vectorstores(data_dir="../data")) 

    pipeline = RAGPipeline(dbs=dbs, mcp_config_path="mcp_config.json") 

    query = "What is the official recommendation regarding X from the report?"
    print(f"\nRunning query: {query}")
    answer = loop.run_until_complete(pipeline.run_async(query)) 
    print("\nFINAL ANSWER:\n", answer)


if __name__ == "__main__":
    main()
    