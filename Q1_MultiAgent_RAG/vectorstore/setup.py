"""
Vector Store - ChromaDB Setup
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class VectorStoreManager:
    """Vector Store Manager using ChromaDB"""
    
    def __init__(self, chroma_path: str, embedding_model: str):
        self.chroma_path = chroma_path
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
    
    def create_vectorstore(self, docs: List[Document]):
        """Create new vector store"""
        os.makedirs(self.chroma_path, exist_ok=True)
        self.vectorstore = Chroma.from_documents(
            docs, self.embeddings, persist_directory=self.chroma_path
        )
        print(f"Vector store created with {self.vectorstore._collection.count()} docs")
    
    def load_vectorstore(self):
        """Load existing vector store"""
        self.vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings
        )
        return self.vectorstore
    
    def get_retriever(self, k: int = 5):
        """Get retriever"""
        if self.vectorstore is None:
            self.load_vectorstore()
        return self.vectorstore.as_retriever(search_kwargs={"k": k})