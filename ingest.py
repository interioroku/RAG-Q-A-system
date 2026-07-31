import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env
load_dotenv()

def load_documents(directory):
    documents = []
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return documents

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isdir(file_path):
            continue
        
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif ext in [".txt", ".md"]:
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    # Splits documents into smaller chunks for retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def run_ingestion(docs_dir="./docs", chunk_size=500, chunk_overlap=100):
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        
    raw_docs = load_documents(docs_dir)
    if not raw_docs:
        return 0
        
    chunks = chunk_documents(raw_docs, chunk_size, chunk_overlap)
    if not chunks:
        return 0
        
    # Embed and persist the chunks in Chroma vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = "./chroma_db"
    
    # Using from_documents recreates the collection with the new chunks
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="rag-collection"
    )
    return len(chunks)

if __name__ == "__main__":
    docs_dir = "./docs"
    print(f"Scanning directory: {docs_dir}")
    num_chunks = run_ingestion(docs_dir)
    if num_chunks == 0:
        print("No documents found or ingested. Please add text, markdown, or PDF files to the docs/ folder.")
    else:
        print(f"Successfully embedded and stored {num_chunks} chunks in './chroma_db'.")
