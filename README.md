# RAG Q&A System

An incremental Retrieval-Augmented Generation (RAG) Q&A system built in Python using LangChain, Chroma, and OpenAI.

![RAG Q&A System - Architecture Diagram](assets/architecture_diagram.jpg)

---

## Current Status: Completed Stage 9

We have completed the RAG pipeline assessment integration using the **Ragas** framework. The Streamlit dashboard now supports both real-time evaluation of chatbot responses and batch benchmark suite execution over preset queries (Faithfulness and Answer Relevancy).

### Project Structure
- `docs/`: Holds the local document library (e.g., Markdown, Text, and PDF files).
- `chroma_db/`: Directory where the local vector database is persisted.
- `assets/`: Stores the project architecture diagrams and assets.
- `ingest.py`: Loads, chunks, embeds, and saves document vectors (modular library & CLI).
- `query.py`: Interactive Q&A loop with conversational memory and follow-up query rephrasing (modular library & CLI).
- `app.py`: Streamlit-based web dashboard providing interactive chat, upload interface, chunk adjustments, real-time retrieval audit logs, and integrated Ragas assessment (real-time + batch benchmark).
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

### 7. Stage 7: Advanced Retrieval
- Combines sparse keyword-based retrieval (BM25) and dense semantic retrieval (Chroma) via Reciprocal Rank Fusion (`EnsembleRetriever`).
- Reranks top candidates using a local cross-encoder model (`ms-marco-MiniLM-L-12-v2` via `flashrank`) to optimize context selection for the generator LLM.
- Presents real-time `Relevance Scores` in the source logs.

### 8. Stage 8: Web UI (Streamlit)
- Integrates a responsive, web-based chat interface using `streamlit`.
- Embeds side-by-side components to separate the conversation from the search audit logs.
- Exposes controls to adjust chunk parameters, upload source documents, and rebuild the Chroma vector database on-the-fly.
- Renders detailed expander audit summaries showing refined queries and raw source chunks with relevance scores.

### 9. Stage 9: RAG Assessment
- Integrates the **Ragas** evaluation framework to measure the quality of the system pipeline.
- Real-time scoring computes `Faithfulness` and `Answer Relevancy` for the latest response on-demand.
- Batch Evaluation Benchmark runs queries against a test suite, generating metrics summaries, tabular breakdowns, and comparative bar charts.
- Employs an LLM-as-a-judge model via GPT-4o-mini to score responses objectively.

---

## How to Run

### Setup Environment
1. Create a virtual environment:
   ```bash
   py -m venv venv
   ```
2. Install dependencies:
   ```bash
   .\venv\Scripts\pip install -r requirements.txt
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

### Step 3: Launch Streamlit Web UI (Optional Dashboard)
Alternatively, start the web interface:
```bash
.\venv\Scripts\streamlit.exe run app.py
```
Open `http://localhost:8501` in your browser. This dashboard lets you upload files, configure chunk parameters, rebuild the database, and chat interactively while inspecting retrieval audit logs.

---

## Next Steps / Future Work

All core building blocks of the incremental RAG pipeline are now complete! Future production-grade enhancements include:
1. **Production Deployment**: Containerizing the setup with Docker and deploying the Streamlit UI to a cloud service.
2. **Context Caching & Semantic Cache**: Adding Redis-based semantic cache to avoid duplicate OpenAI generation costs for identical queries.
