# RAG Q&A System

An incremental Retrieval-Augmented Generation (RAG) Q&A system built in Python using LangChain, Chroma, and OpenAI.

---

## Current Status: Completed Stage 6

We have completed the core command-line RAG pipeline with conversational memory, allowing for multi-turn dialogues and follow-up question refinement.

### Project Structure
- `docs/`: Holds the local document library (e.g., Markdown, Text, and PDF files).
- `chroma_db/`: Directory where the local vector database is persisted.
- `ingest.py`: Loads, chunks, embeds, and saves document vectors.
- `query.py`: Interactive Q&A loop with conversational memory and follow-up query rephrasing.
- `.env`: Configures API keys and tracing settings (ignored by Git).
- `.gitignore`: Ensures venv, cache, database folders, and credentials are not committed.

---

## Completed Build Stages

### 1. Stage 1: Document Loading & Chunking
- Scans `docs/` for `.md`, `.txt`, and `.pdf` files.
- Uses `RecursiveCharacterTextSplitter` to split files into smaller, manageable chunks.

### 2. Stage 2: Embedding & Vector Store
- Integrates `OpenAIEmbeddings` (`text-embedding-3-small`).
- Configures local `Chroma` database and stores document chunks in a collection named `rag-collection`.

### 3. Stage 3: Retrieval
- Performs semantic similarity search on the vector store.
- Prints matches along with their metadata source and similarity distance scores.

### 4. Stage 4: Generation
- Employs `ChatOpenAI` (`gpt-4o-mini`) to generate grounded responses.
- Prompt constraints instruct the LLM to output "I don't know" rather than hallucinating when relevant context is missing.

### 5. Stage 5: Q&A Loop
- Wraps the search and generation steps in an interactive terminal interface.

### 6. Stage 6: Conversational Memory
- Adds structured history logs using `HumanMessage` and `AIMessage`.
- Automatically refines follow-up queries (e.g. converting *"What can it be spent on?"* to *"What types of expenses can the employee training budget be used for?"*) before querying the database.

---

## How to Run

### Setup Environment
1. Create a virtual environment:
   ```bash
   py -m venv venv
   ```
2. Install dependencies:
   ```bash
   .\venv\Scripts\pip install langchain langchain-community langchain-text-splitters pypdf python-dotenv langchain-openai langchain-chroma
   ```
3. Create a `.env` file at the root containing your OpenAI API Key:
   ```env
   OPENAI_API_KEY=your-api-key-here
   ```

### Step 1: Ingest Documents
Place your documentation (PDF/Text/Markdown) in the `docs/` folder, then run:
```bash
.\venv\Scripts\python.exe ingest.py
```
This splits and writes the document chunks to `./chroma_db`.

### Step 2: Query / Start Q&A Session
Run the interactive loop:
```bash
.\venv\Scripts\python.exe query.py
```
Type your questions. Type `exit` or `quit` to end the session.

---

## Next Steps / Future Work
When returning to this project, the next planned enhancements are:
1. **Stage 7: Advanced Retrieval**
   - Implement hybrid keyword search (BM25) combined with semantic vector search.
   - Integrate cross-encoder reranking (e.g., Cohere/HuggingFace) to improve retrieval precision.
2. **Stage 8: Web UI (Streamlit)**
   - Develop a visual dashboard for uploading documents, adjusting chunk parameters, displaying retrieval logs, and conversing with the system in a browser UI.
3. **Stage 9: RAG Assessment**
   - Add RAG evaluation frameworks (e.g., Ragas or TruLens) to measure faithfulness, answer relevance, and context recall.
