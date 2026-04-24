"""
Multi-Agent RAG System - Main Entry Point
Run: python main.py
"""

import sys
import asyncio
from config import Config
from utils.data_loader import PDFDataLoader, DocumentChunker
from vectorstore.setup import VectorStoreManager
from agents import RetrieverAgent, GraderAgent, GeneratorAgent, LLMClient


class MultiAgentRAG:
    """Multi-Agent RAG System"""
    
    def __init__(self):
        self.vs_manager = None
        self.llm_client = None
    
    def initialize(self):
        """Initialize the RAG system"""
        print("Initializing RAG system...")
        
        pdf_loader = PDFDataLoader(str(Config.DATA_PATH))
        docs = pdf_loader.load_all_pdfs()
        
        chunker = DocumentChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        chunks = chunker.chunk_documents(docs)
        
        self.vs_manager = VectorStoreManager(str(Config.CHROMA_PATH), Config.EMBEDDING_MODEL)
        self.vs_manager.create_vectorstore(chunks)
        
        self.llm_client = LLMClient(Config.API_KEY, Config.MODEL_NAME)
        
        print("Ready!")
    
    def query(self, user_query: str) -> dict:
        """Process a user query"""
        retriever = self.vs_manager.get_retriever(k=Config.TOP_K)
        
        rag = RetrieverAgent(retriever)
        result = asyncio.run(rag.retrieve(user_query, k=Config.TOP_K))
        
        doc_texts = [d.page_content for d in result.documents]
        
        grader = GraderAgent(self.llm_client)
        grade = grader.grade(user_query, doc_texts)
        
        if grade.is_relevant:
            generator = GeneratorAgent(self.llm_client)
            gen = generator.generate(user_query, doc_texts)
            return {
                "answer": gen.answer,
                "sources": gen.sources,
                "relevance": grade.relevance_score
            }
        
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "relevance": grade.relevance_score
        }


def main():
    """Interactive CLI"""
    print("=" * 50)
    print("  Multi-Agent RAG System")
    print("=" * 50)
    print("\nStarting...")
    
    rag = MultiAgentRAG()
    rag.initialize()
    
    print("\n" + "=" * 50)
    print("  READY - Ask your questions!")
    print("=" * 50)
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            q = input("You: ").strip()
            if not q or q.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            result = rag.query(q)
            print("\nANSWER:", result['answer'])
            print("Sources:", result['sources'])
            print("Relevance:", str(result['relevance']) + "/10\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()