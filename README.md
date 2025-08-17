# Naive RAG Chatbot

Lightweight Retrieval-Augmented Generation (RAG) demo focused on health/disease Q&A. The app lets you ingest local documents (PDF, TXT, CSV) and web pages, index them with FAISS, and ask questions or request summaries over the ingested content.

Core features
- Document ingestion: PDF, TXT, CSV, and URL scraping (BeautifulSoup-based).
- Vector store: FAISS (in-memory + optional on-disk persistence `faiss_index`).
- Reranker: optional Cross-Encoder reranker (sentence-transformers CrossEncoder) to improve retrieval quality.
- UI: Streamlit app with file upload, URL ingestion, Q&A and Summarization modes.
- LLM backends: Groq (via langchain-groq) if `GROQ_API_KEY` is set


Quick start
1. Create and activate a virtual environment (Python 3.10+ recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
3. (Optional) Add API keys to environment or a `.env` file in this folder:
```bash
export GROQ_API_KEY="your_groq_key"
```
4. Run the Streamlit app:
```bash
streamlit run app.py
```

How to use the app
1. Open the app in your browser (Streamlit prints the local URL).
2. In the left sidebar:
	- Paste a URL to ingest a web article, or upload files using the uploader.
	- Optionally check "Persist FAISS index to disk" to save the index to `faiss_index`.
	- Toggle "Use Cross-Encoder reranker" to enable reranking after retrieval.
	- Choose mode: `Q&A` or `Summarize`.
	- Click `Ingest` to process documents and build/update the index.
3. Enter a question or summarization instructions and click `Get Answer`.

Persistence and index behavior
- When persistence is enabled, the index is stored in the `faiss_index` directory.
- On startup the app will attempt to auto-load `faiss_index` if present. If an index exists it will be loaded and new ingested documents will be added to it.
- Security note: FAISS index loading uses Python pickle under the hood. The app sets `allow_dangerous_deserialization=True` when loading the index to allow reloading local indexes; do not load `faiss_index` folders from untrusted sources.

LLM backends
- Groq: if `GROQ_API_KEY` is present and `langchain-groq` installed, the app will use Groq via `ChatGroq`.
- The app tries a few invocation patterns (.invoke, direct call, .generate) and extracts text from common response shapes.

Example queries
- "What is diabetic retinopathy?"
- "Summarize the risk factors and recommended screening intervals for diabetic retinopathy."
- "List key treatment options for advanced diabetic retinopathy from the documents."

Troubleshooting
- If you see "Index is empty — ingest local files or provide a URL in the sidebar": either you didn't persist the index when ingesting, or the app failed to load the persisted index. Check the app logs and ensure `faiss_index` exists.
- If you see a pickle-related ValueError when loading `faiss_index`, the app now loads with `allow_dangerous_deserialization=True` but you should only do this for indexes you created/trust. To recreate a fresh index:
```bash
rm -rf faiss_index
# then ingest again via the app or CLI
```

Known limitations
- The summarization and reranker require additional packages (CrossEncoder models, heavy HF models) and may download large models on first run.
- FAISS persistence uses pickle; consider alternatives for production.


Here is the link for the live demo : https://naive-rag-chatbot.streamlit.app/ (😅 wake up might take a little time...)


