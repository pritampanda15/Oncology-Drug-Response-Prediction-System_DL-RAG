"""
src/rag/ingestion.py

Document loading and chunking for the knowledge base.
"""

from pathlib import Path
from typing import Optional


class Document:
    
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata
    
    def __repr__(self):
        return f"Document(source={self.metadata.get('source', 'unknown')}, chars={len(self.content)})"


class KnowledgeBaseLoader:
    
    def __init__(self, documents_dir: str):
        self.documents_dir = Path(documents_dir)
    
    def load_documents(self) -> list[Document]:
        documents = []
        
        for filepath in self.documents_dir.glob("*.txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            drug_name = filepath.stem.replace("_", " ").title()
            
            doc = Document(
                content=content,
                metadata={
                    "source": filepath.name,
                    "drug_name": drug_name,
                    "filepath": str(filepath)
                }
            )
            documents.append(doc)
            print(f"Loaded: {filepath.name} ({len(content)} chars)")
        
        return documents
    
    def chunk_document(
        self, 
        document: Document, 
        chunk_size: int = 500,
        overlap: int = 50
    ) -> list[Document]:
        """
        Split document into smaller chunks for embedding.
        Tries to split on paragraph boundaries when possible.
        """
        content = document.content
        paragraphs = content.split("\n\n")
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                content=chunk,
                metadata={
                    **document.metadata,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            chunk_documents.append(chunk_doc)
        
        return chunk_documents
    
    def load_and_chunk_all(self, chunk_size: int = 500) -> list[Document]:
        documents = self.load_documents()
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc, chunk_size=chunk_size)
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks from {doc.metadata['source']}")
        
        print(f"\nTotal chunks: {len(all_chunks)}")
        return all_chunks
