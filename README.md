# 🤖 OmniDoc-RAG-Agent
### *Advanced Industrial Document Intelligence Pipeline*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Powered%20By-LangChain-green.svg)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**OmniDoc-RAG-Agent** is a sophisticated Retrieval-Augmented Generation (RAG) system designed for high-stakes industrial environments. It goes beyond simple document search by implementing **Semantic Chunking**, **Hybrid Vector Retrieval**, and **Self-Correction Loops** to ensure maximum accuracy and reliability.

## 🌟 Key Features
- **Semantic Document Processing**: Uses advanced embedding models to split documents based on meaning rather than fixed character counts.
- **Hybrid Search Engine**: Combines Dense Vector Search (ChromaDB) with Keyword Matching for robust retrieval.
- **Industrial-Grade Reliability**: Built-in exception handling for complex PDF structures and corrupt metadata.
- **Multi-LLM Integration**: Seamlessly switch between OpenAI (GPT-4), Anthropic (Claude), or Local Models (Llama-3 via Ollama).
- **Asynchronous Pipeline**: Optimized for performance with async document processing.

## 🏗️ Architecture
`mermaid
graph TD
    A[Raw Documents] --> B[Smart Loader]
    B --> C[Semantic Chunker]
    C --> D[Vector Store - ChromaDB]
    D --> E[Hybrid Retriever]
    F[User Query] --> E
    E --> G[Context Re-ranker]
    G --> H[LLM - Reasoning Engine]
    H --> I[Verified Answer]
`

## 🚀 Quick Start
1. **Clone the Repo**
   `ash
   git clone https://github.com/fadhel-haidar/OmniDoc-RAG-Agent.git
   cd OmniDoc-RAG-Agent
   `

2. **Setup Environment**
   `ash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate
   pip install -r requirements.txt
   `

3. **Configure API Keys**
   Create a .env file:
   `env
   OPENAI_API_KEY=your_key_here
   `

4. **Run the Agent**
   `ash
   python main.py --path ./data/manual_unit_A1.pdf
   `

---

## 🛠️ Tech Stack
- **Framework**: LangChain
- **Embeddings**: HuggingFace (Local) or OpenAI
- **Vector DB**: ChromaDB
- **Models**: GPT-4 / Claude 3 / Llama 3

## 🧑‍💻 About the Author
Developed by **Fadhel Haidar**, an AI Engineer specializing in RAG systems and Computer Vision at **PT United Tractors Tbk**. Focused on bridging the gap between LLMs and industrial operational efficiency.

---
*This repository is for educational and professional demonstration purposes.*