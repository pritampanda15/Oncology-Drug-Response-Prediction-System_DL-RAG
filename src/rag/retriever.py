"""
src/rag/retriever.py

Vector store and semantic search for RAG.
Uses sentence-transformers for local embeddings.
"""

from pathlib import Path
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions

from src.rag.ingestion import Document, KnowledgeBaseLoader


class KnowledgeBaseRetriever:
    
    def __init__(
        self, 
        persist_directory: str = "knowledge_base/vector_store",
        collection_name: str = "drug_knowledge"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "Drug response knowledge base"}
        )
    
    def index_documents(self, documents: list[Document]) -> None:
        if len(documents) == 0:
            print("No documents to index")
            return
        
        ids = []
        contents = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            doc_id = f"{doc.metadata.get('source', 'doc')}_{doc.metadata.get('chunk_id', i)}"
            ids.append(doc_id)
            contents.append(doc.content)
            metadatas.append(doc.metadata)
        
        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas
        )
        
        print(f"Indexed {len(documents)} document chunks")
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        drug_filter: Optional[str] = None
    ) -> list[dict]:
        
        where_filter = None
        if drug_filter:
            where_filter = {"drug_name": drug_filter}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results["distances"] else None
            })
        
        return search_results
    
    def get_collection_stats(self) -> dict:
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name
        }


def build_knowledge_base(documents_dir: str, vector_store_dir: str) -> KnowledgeBaseRetriever:
    loader = KnowledgeBaseLoader(documents_dir)
    chunks = loader.load_and_chunk_all(chunk_size=500)
    
    retriever = KnowledgeBaseRetriever(persist_directory=vector_store_dir)
    retriever.index_documents(chunks)
    
    return retriever