"""
scripts/build_knowledge_base.py

Build the vector store from knowledge base documents.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.retriever import build_knowledge_base


def main():
    print("Building knowledge base...\n")
    
    retriever = build_knowledge_base(
        documents_dir="knowledge_base/documents",
        vector_store_dir="knowledge_base/vector_store"
    )
    
    stats = retriever.get_collection_stats()
    print(f"\nKnowledge base ready:")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Documents: {stats['total_documents']}")
    
    print("\nTesting search...")
    results = retriever.search("cisplatin resistance BRCA1", n_results=2)
    for r in results:
        print(f"\n[{r['metadata'].get('source')}]")
        print(f"{r['content'][:200]}...")


if __name__ == "__main__":
    main()
