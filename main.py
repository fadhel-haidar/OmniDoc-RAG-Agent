import argparse
import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from core.engine import RAGCore

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="OmniDoc-RAG-Agent Entry Point")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file for ingestion")
    parser.add_argument("--query", type=str, help="Question to ask the AI")
    args = parser.parse_args()

    # Safety check for API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment. Please check your .env file.")
        return

    rag = RAGCore(api_key=api_key)

    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"❌ Error: File {args.pdf} not found.")
            return
            
        print(f"📂 Processing document: {args.pdf}...")
        loader = PyPDFLoader(args.pdf)
        docs = loader.load()
        rag.ingest_documents(docs)
        print("✅ Ingestion complete.")

    if args.query:
        print(f"💬 Query: {args.query}")
        try:
            answer = rag.query(args.query)
            print(f"\n🤖 AI Answer:\n{answer}")
        except Exception as e:
            print(f"❌ Error during query: {str(e)}")

if __name__ == "__main__":
    main()