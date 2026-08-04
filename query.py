import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document

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

def initialize_retriever(vector_store, k=5):
    if vector_store is None:
        return None
        
    # Retrieve all documents to fit the BM25 sparse retriever
    db_data = vector_store.get()
    docs = []
    if db_data and "documents" in db_data:
        metadatas = db_data.get("metadatas") or [{}] * len(db_data["documents"])
        for text, metadata in zip(db_data["documents"], metadatas):
            docs.append(Document(page_content=text, metadata=metadata))
            
    # Chroma dense retriever
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    if not docs:
        print("Warning: Vector store contains no documents. Falling back to vector search only.")
        return chroma_retriever

    # BM25 sparse retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k
    
    # Reciprocal Rank Fusion ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6]
    )
    
    # Local cross-encoder reranker
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=2)
    
    # Wrapped compressed retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever

def retrieve_chunks(retriever, query):
    if retriever is None:
        return []
    results = retriever.invoke(query)
    return results

def rephrase_question(llm, query, chat_history):
    if not chat_history:
        return query
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )),
        *chat_history,
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"question": query})
    return response.content.strip()

def generate_answer(llm, query, retrieved_docs, chat_history):
    # Combine content of retrieved docs as context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an assistant for question-answering tasks.\n"
            "Use the following pieces of retrieved context and conversation history to answer the question.\n"
            "If you don't know the answer, say 'I don't know'. Do not try to make up an answer.\n"
            "Keep the answer concise and factual.\n\n"
            "Context:\n{context}"
        )),
        *chat_history,
        ("human", "{question}")
    ])
    
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
        
    print("Initializing Hybrid Retriever and Reranker (this might take a moment on first load)...")
    retriever = initialize_retriever(vector_store, k=5)
    
    # Initialize the LLM (gpt-4o-mini is efficient and fast)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Initialize conversational chat history
    chat_history = []
        
    print("\n" + "=" * 60)
    print("Interactive RAG Q&A System Active (with Hybrid Search & Reranking)!")
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
            
        # Rephrase query using chat history if follow-up
        rephrased_query = query
        if chat_history:
            rephrased_query = rephrase_question(llm, query, chat_history)
            if rephrased_query != query:
                print(f"[Refined Query: '{rephrased_query}']")
            
        print("Retrieving context...")
        results = retrieve_chunks(retriever, rephrased_query)
        
        if not results:
            print("No matching context chunks found in database.")
            continue
            
        print("Generating answer...")
        answer = generate_answer(llm, query, results, chat_history)
        
        print("\n--- Answer ---")
        print(answer)
        print("\n--- Source Chunks Used ---")
        for idx, doc in enumerate(results, 1):
            score = doc.metadata.get('relevance_score')
            score_str = f"{score:.4f}" if score is not None else "N/A"
            page = doc.metadata.get('page')
            page_str = f" (Page {page + 1})" if page is not None else ""
            print(f"[{idx}] Source: {doc.metadata.get('source')}{page_str} (Relevance Score: {score_str})")
        print("\n" + "-" * 60 + "\n")
        
        # Append to chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
