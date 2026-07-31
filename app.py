import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Import existing modular functions to preserve repository design patterns
from ingest import run_ingestion
from query import load_vector_store, initialize_retriever, retrieve_chunks, rephrase_question, generate_answer

# Load environmental configurations
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f121d 0%, #16192b 100%);
        color: #fafafa;
    }
    
    /* Sidebar styling override */
    [data-testid="stSidebar"] {
        background-color: #0c0e18 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Header card */
    .main-header {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* Custom button aesthetics */
    div.stButton > button {
        background: linear-gradient(45deg, #4f46e5, #06b6d4) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.5) !important;
    }
    div.stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Expander card layout */
    .st-ae {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="main-header">
    <h1 style='margin: 0; font-weight: 700; background: linear-gradient(45deg, #818cf8, #22d3ee); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        📊 Advanced RAG Q&A Dashboard
    </h1>
    <p style='margin: 10px 0 0 0; font-size: 16px; color: #9ca3af;'>
        Hybrid Search (BM25 + Chroma) & Local Cross-Encoder Reranking (Flashrank)
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_rephrased_query" not in st.session_state:
    st.session_state.latest_rephrased_query = None
if "latest_retrieval_results" not in st.session_state:
    st.session_state.latest_retrieval_results = None

# Sidebar Controls
st.sidebar.markdown("### ⚙️ System Controls")

# API Status Verification
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.sidebar.error("⚠️ OPENAI_API_KEY missing in environment!")
else:
    st.sidebar.success("✅ OpenAI API Configured")

# 1. Document Manager Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Document Uploader")
uploaded_files = st.sidebar.file_uploader(
    "Choose files to add to library:",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("./docs", exist_ok=True)
    save_success = False
    for f in uploaded_files:
        file_path = os.path.join("./docs", f.name)
        # Avoid saving duplicate files
        if not os.path.exists(file_path):
            with open(file_path, "wb") as buffer:
                buffer.write(f.getbuffer())
            save_success = True
            
    if save_success:
        st.sidebar.info("Files saved in `./docs`. Rebuild database to index them.")

# 2. Ingestion Hyperparameters
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Ingestion Settings")
chunk_size = st.sidebar.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=500, step=50)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=50, max_value=500, value=100, step=10)

if st.sidebar.button("🔄 Rebuild Vector Store"):
    with st.spinner("Processing documents, generating embeddings & rebuilding Chroma database..."):
        try:
            num_chunks = run_ingestion(docs_dir="./docs", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if num_chunks > 0:
                st.sidebar.success(f"Success! Index built with {num_chunks} chunks.")
                st.rerun()
            else:
                st.sidebar.warning("No files found to ingest inside `./docs` folder.")
        except Exception as e:
            st.sidebar.error(f"Ingestion error: {e}")

# 3. Search Tuning
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Retrieval Controls")
retrieval_k = st.sidebar.slider("Candidate Retrieve Count (K)", min_value=2, max_value=15, value=5)

# Reset Button
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Reset Conversational Memory"):
    st.session_state.chat_history = []
    st.session_state.latest_rephrased_query = None
    st.session_state.latest_retrieval_results = None
    st.sidebar.success("Memory cleared!")
    st.rerun()

# Database Check
vector_store = load_vector_store()
if vector_store is None:
    st.warning("⚠️ Local Chroma Database not detected. Please upload documents in the sidebar and click **🔄 Rebuild Vector Store**.")
else:
    # Initialize main interface layout (Columns)
    col1, col2 = st.columns([1.2, 0.8], gap="medium")

    # Column 1: Conversational Chat Interface
    with col1:
        st.markdown("### 💬 Conversational Chat")
        
        # Display chat logs
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
                    
        # Listen to user query
        if prompt := st.chat_input("Ask a question about your stored documentation..."):
            # Display user message instantly
            with st.chat_message("user"):
                st.write(prompt)
                
            # Initialize core models
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            # 1. Standalone Query Refinement (Conversational memory)
            rephrased_query = prompt
            if st.session_state.chat_history:
                with st.spinner("Refining follow-up query..."):
                    rephrased_query = rephrase_question(llm, prompt, st.session_state.chat_history)
            st.session_state.latest_rephrased_query = rephrased_query
            
            # 2. Retrieve Candidate Chunks (Hybrid Search + Reranking)
            retriever = initialize_retriever(vector_store, k=retrieval_k)
            with st.spinner("Performing hybrid search and cross-encoder reranking..."):
                results = retrieve_chunks(retriever, rephrased_query)
            st.session_state.latest_retrieval_results = results
            
            # 3. Grounded Answer Generation
            if results:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(llm, prompt, results, st.session_state.chat_history)
            else:
                answer = "I don't know (no matching context chunks found in database)."
                
            with st.chat_message("assistant"):
                st.write(answer)
                
            # Append interaction to dialogue logs
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=answer))
            st.rerun()

    # Column 2: Audit Logs Panel
    with col2:
        st.markdown("### 🔍 Retrieval Audit Logs")
        
        if st.session_state.latest_rephrased_query:
            st.info(f"**Last Refined Query Sent to Retriever:**\n`{st.session_state.latest_rephrased_query}`")
            
            results = st.session_state.latest_retrieval_results
            if results:
                st.markdown(f"**Retrieved Sources (Top {len(results)} after Flashrank Reranking):**")
                for idx, doc in enumerate(results, 1):
                    score = doc.metadata.get('relevance_score')
                    score_val = f"{score:.4f}" if score is not None else "N/A"
                    source_name = doc.metadata.get('source', 'Unknown')
                    
                    with st.expander(f"Chunk #{idx} | Relevance Score: {score_val} | File: {os.path.basename(source_name)}"):
                        st.markdown(f"**Relevance Score:** `{score_val}`")
                        st.markdown(f"**Source File Path:** `{source_name}`")
                        st.markdown("**Chunk Content:**")
                        st.code(doc.page_content, language="markdown")
            else:
                st.warning("No matching source chunks found in the database.")
        else:
            st.markdown(
                "<div style='border: 1px dashed rgba(255,255,255,0.15); border-radius: 8px; padding: 40px; text-align: center; color: #6b7280;'>"
                "Ask a question in the chat to populate search logs."
                "</div>",
                unsafe_allow_html=True
            )
