"""
Compatibility wrapper for rag_pipeline module

The original file had a bug and a misspelled filename. Keep this wrapper so
older references still work; delegate to the fixed implementation in
`rag_pipeline.py`.
"""

from .rag_pipeline import RAGPipeline  # re-export the fixed implementation

__all__ = ["RAGPipeline"]