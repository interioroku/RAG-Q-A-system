import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env
load_dotenv()

def retrieve_chunks(query, k=2):
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        print(f"Error: Vector store directory '{persist_directory}' does not exist. Please run ingest.py first.")
        return []
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load the persisted vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="rag-collection"
    )
    
    # Perform similarity search with distance score (lower score = higher similarity)
    results = vector_store.similarity_search_with_score(query, k=k)
    return results

if __name__ == "__main__":
    test_query = "What is the training budget for employees?"
    print(f"Query: '{test_query}'\n")
    print("Retrieving relevant chunks from Chroma...")
    
    results = retrieve_chunks(test_query, k=2)
    
    if not results:
        print("No matches found.")
    else:
        print(f"Retrieved {len(results)} chunks:")
        for idx, (doc, score) in enumerate(results, 1):
            print(f"\n--- Result {idx} (Distance Score: {score:.4f}) ---")
            print(f"Source: {doc.metadata.get('source')}")
            print(f"Content:\n{doc.page_content}")
            print("-" * 40)
