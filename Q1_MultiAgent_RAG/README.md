# Multi-Agent RAG Orchestration - Virallens Assignment

## Overview
This project implements a multi-agent RAG (Retrieval-Augmented Generation) workflow using **LangChain**, **LangGraph**, and **Groq API**.

## Agent Roles

### 1. Retriever Agent
- **Role**: Searches and retrieves relevant documents from the vector store
- **Inputs**: User query
- **Outputs**: Top-k most relevant document chunks with relevance scores

### 2. Grader Agent  
- **Role**: Evaluates the relevance of retrieved documents to the user query
- **Inputs**: User query + Retrieved documents
- **Outputs**: Relevance score (0-10), yes/no decision, reasoning

### 3. Generator Agent
- **Role**: Generates final natural language answer based on relevant documents
- **Inputs**: User query + Graded relevant documents
- **Outputs**: Final answer with source attribution

## LangGraph Flow

```
Query -> [Retrieve] -> Documents -> [Grade] -> Relevance Check
                                              |
                                         Is Relevant?
                                              |
                          +------------------+----+------------------+
                          |                                     |
                    [Generate]                             [Transform]
                          |                                     |
                          +------------------+------------------+
                                           |
                                      [Retrieve]
                                           |
                                         [END]
```

## Technical Implementation

### Chunking Methodology
- **Recursive Character Text Splitter**
- Chunk size: 1000 characters
- Overlap: 200 characters
- Separators: ["\n\n", "\n", ". ", " ", ""]

### API Integration
- **Primary**: Groq API (llama-3.3-70b-versatile)
- **Fallback**: Rule-based responses when API unavailable
- Rate limiting handled with retry logic (3 retries, 2s delay)

### Vector Store
- ChromaDB with sentence-transformers/all-MiniLM-L6-v2 embeddings
- 384-dimensional embeddings
- Persisted to disk

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` file:
```
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

## Running

```bash
# Run demo with sample queries
python demo.py

# Run your own queries
python test_query.py
```

## Docker

```bash
docker build -t virallens-rag .
docker run -e GROQ_API_KEY=your_key virallens-rag
```

## Project Structure

```
Q1_MultiAgent_RAG/
├── agents/
│   ├── retriever_agent.py    # Document retrieval
│   ├── grader_agent.py   # Relevance grading
│   └── generator_agent.py  # Answer generation
├── graph/
│   └── workflow.py        # LangGraph orchestration
├── vectorstore/
│   └── setup.py          # ChromaDB setup
├── utils/
│   └── data_loader.py   # PDF loading & chunking
├── data/                 # PDF documents
├── main.py              # Entry point
├── config.py           # Configuration
├── Dockerfile
└── README.md
```

## Deliverables Checklist

| Requirement | Status |
|-------------|--------|
| Runnable repo | ✓ |
| Dockerfile | ✓ |
| Agent roles documented | ✓ |
| LangGraph flow | ✓ |
| Chunking methodology | ✓ |
| Free API integration | ✓ (Groq) |
| Rate limiting | ✓ |
| README | ✓ |