import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

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
    queries = [
        "What is the training budget for employees?",
        "What is the company's policy on office pets?"
    ]
    
    for q in queries:
        print("=" * 60)
        print(f"Query: '{q}'")
        print("=" * 60)
        print("Retrieving relevant chunks from Chroma...")
        results = retrieve_chunks(q, k=2)
        
        if not results:
            print("No matching chunks found in database.")
            continue
            
        print(f"Retrieved {len(results)} chunks. Generating answer...")
        answer = generate_answer(q, results)
        
        print("\n--- Answer ---")
        print(answer)
        print("\n--- Source Chunks Used ---")
        for idx, (doc, score) in enumerate(results, 1):
            print(f"[{idx}] Source: {doc.metadata.get('source')} (Distance: {score:.4f})")
        print("\n")
