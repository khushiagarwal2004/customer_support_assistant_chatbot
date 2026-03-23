from rag_engine import RAGEngine

if __name__ == "__main__":
    print("🚀 Starting manual knowledge base ingestion...")
    engine = RAGEngine()
    
    # We call rebuild() rather than _ingest_knowledge_base() 
    # to ensure a fresh clean ingestion into the ChromaDB vector store
    engine.rebuild()
    
    print("✅ Ingestion fully completed.")
    print("You can now start the application with 'python app.py'")
