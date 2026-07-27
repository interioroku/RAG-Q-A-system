import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

def load_vector_store():
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        print(f"Error: Vector store directory '{persist_directory}' does not exist. Please run ingest.py first.")
        return None
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load the persisted vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="rag-collection"
    )
    return vector_store

def retrieve_chunks(vector_store, query, k=2):
    if vector_store is None:
        return []
    # Perform similarity search with distance score (lower score = higher similarity)
    results = vector_store.similarity_search_with_score(query, k=k)
    return results

def generate_answer(query, retrieved_docs):
    # Combine content of retrieved docs as context
    context = "\n\n".join([doc.page_content for doc, _ in retrieved_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an assistant for question-answering tasks.\n"
            "Use the following pieces of retrieved context to answer the question.\n"
            "If you don't know the answer, say 'I don't know'. Do not try to make up an answer.\n"
            "Keep the answer concise and factual.\n\n"
            "Context:\n{context}"
        )),
        ("human", "{question}")
    ])
    
    # Initialize the LLM (gpt-4o-mini is efficient and fast)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = prompt | llm
    
    response = chain.invoke({
        "context": context,
        "question": query
    })
    
    return response.content

if __name__ == "__main__":
    print("Loading vector database...")
    vector_store = load_vector_store()
    if vector_store is None:
        exit(1)
        
    print("\n" + "=" * 60)
    print("Interactive RAG Q&A System Active!")
    print("Type your questions below. Type 'exit' or 'quit' to end.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            query = input("Ask a question: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break
            
        query = query.strip()
        if not query:
            continue
            
        if query.lower() in ["exit", "quit"]:
            print("Exiting. Goodbye!")
            break
            
        print("Retrieving context...")
        results = retrieve_chunks(vector_store, query, k=2)
        
        if not results:
            print("No matching context chunks found in database.")
            continue
            
        print("Generating answer...")
        answer = generate_answer(query, results)
        
        print("\n--- Answer ---")
        print(answer)
        print("\n--- Source Chunks Used ---")
        for idx, (doc, score) in enumerate(results, 1):
            print(f"[{idx}] Source: {doc.metadata.get('source')} (Distance: {score:.4f})")
        print("\n" + "-" * 60 + "\n")
