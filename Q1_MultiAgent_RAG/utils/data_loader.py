"""
Data Loader - PDF Loading and Chunking
"""

import os
from typing import List
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdfs(data_path: str) -> List[Document]:
    """Load all PDFs from data directory"""
    docs = []
    for f in os.listdir(data_path):
        if f.endswith('.pdf'):
            path = os.path.join(data_path, f)
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={'source': f, 'page': i + 1}
                    ))
    return docs


def chunk_documents(docs: List[Document], chunk_size: int = 1000, overlap: int = 200) -> List[Document]:
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)


class PDFDataLoader:
    """PDF Data Loader"""
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_all_pdfs(self) -> List[Document]:
        return load_pdfs(self.data_path)


class DocumentChunker:
    """Document Chunking"""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        return chunk_documents(docs, self.chunk_size, self.chunk_overlap)