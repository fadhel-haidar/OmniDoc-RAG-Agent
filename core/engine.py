import logging
import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Configure logging for better observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OmniDoc-RAG")

class RAGCore:
    def __init__(self, api_key: str, model_name: str = "gpt-4-turbo-preview"):
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key, temperature=0.1)
        self.vector_store = None
        logger.info(f"Initialized RAGCore with model: {model_name}")

    def ingest_documents(self, documents: List[Document], persist_directory: str = "./db"):
        \"\"\"
        Processes and indexes documents into the vector store.
        
        Args:
            documents: List of LangChain Document objects.
            persist_directory: Path to store ChromaDB data.
        \"\"\"
        logger.info(f"Ingesting {len(documents)} documents...")
        
        # We use RecursiveCharacterTextSplitter to maintain semantic flow
        # Overlap is crucial to ensure context isn't lost between chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks.")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        self.vector_store.persist()
        logger.info("Vector store persisted successfully.")

    def query(self, user_question: str) -> str:
        \"\"\"
        Standard RAG query pipeline.
        
        Args:
            user_question: The query from user.
        \"\"\"
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please ingest documents first.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        response = qa_chain.invoke({"query": user_question})
        return response["result"]

# For demonstration: How we handle 'Semantic Chunking' or other advanced logic
# This is a placeholder for Fadhel to expand upon as needed for his industrial use cases.