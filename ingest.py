import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def chunk_documents(documents):
    # Splits documents into smaller chunks for retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

if __name__ == "__main__":
    docs_dir = "./docs"
    print(f"Scanning directory: {docs_dir}")
    raw_docs = load_documents(docs_dir)
    print(f"Loaded {len(raw_docs)} document files/pages.")
    
    if not raw_docs:
        print("No documents found. Please add text, markdown, or PDF files to the docs/ folder.")
    else:
        chunks = chunk_documents(raw_docs)
        print(f"Created {len(chunks)} chunks.")
        if chunks:
            print("\n--- Sample Chunk ---")
            print(f"Source: {chunks[0].metadata.get('source')}")
            print(f"Content Preview:\n{chunks[0].page_content}")
            print("--------------------")
