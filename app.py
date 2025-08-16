import streamlit as st
import os
import sys
import pathlib
from dotenv import load_dotenv

# allow running from repo root
sys.path.append(str(pathlib.Path(__file__).parent))

from ingest_and_index import ingest_paths
from reranker import rerank
from langchain.prompts import PromptTemplate
from utils import path_to_loader

# LLM backends: prefer Groq if GROQ_API_KEY set
import os
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if GROQ_API_KEY:
    # print("Using Groq API")
    try:
        from langchain_groq import ChatGroq
        def make_llm(**kwargs):
            # print("let make the LLM (lol)")
            return ChatGroq(model_name=kwargs.get('model_name', 'llama-3.3-70b-versatile'), temperature=kwargs.get('temperature', 0))
    except Exception:
        make_llm = None

load_dotenv()

st.set_page_config(page_title="Naive RAG Health Chatbot", page_icon="🩺", layout="wide")
st.title("Naive RAG Health Chatbot")

if 'vectorstore' not in st.session_state:
    # try auto-load persisted FAISS index if exists
    if os.path.exists('faiss_index'):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            st.session_state.vectorstore = FAISS.load_local('faiss_index', emb, allow_dangerous_deserialization=True)
            st.sidebar.success('Loaded persisted FAISS index')
            st.sidebar.warning('Loaded persisted FAISS index using pickle deserialization. Only do this with indexes you trust.')
        except Exception:
            st.session_state.vectorstore = None
    else:
        st.session_state.vectorstore = None

st.sidebar.header('Data')
url = st.sidebar.text_input('Or supply a URL to fetch')
upload = st.sidebar.file_uploader('Upload files', accept_multiple_files=True)
persist = st.sidebar.checkbox('Persist FAISS index to disk', value=False)
use_reranker = st.sidebar.checkbox('Use Cross-Encoder reranker', value=False)
mode = st.sidebar.selectbox('Mode', ['Q&A', 'Summarize'])

if st.sidebar.button('Ingest'):
    paths = []
    if url:
        paths.append(url)
    # handle uploaded files: save to temp and add path
    if upload:
        import tempfile
        for up in upload:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix='_'+up.name)
            tf.write(up.read())
            tf.close()
            paths.append(tf.name)

    if not paths:
        st.sidebar.warning('Add at least one local file path, URL, or upload a file')
    else:
        with st.spinner('Ingesting...'):
            vs = ingest_paths(paths, persist=persist, index_path='faiss_index')
            st.session_state.vectorstore = vs
            st.sidebar.success('Ingestion complete')

if st.session_state.vectorstore is None:
    st.info('Index is empty — ingest local files or provide a URL in the sidebar')
else:
    if mode == 'Q&A':
        question = st.text_input('Ask a health-related question:')
    else:
        question = st.text_area('Summarization instructions:', value='')
    if st.button('Get Answer'):
        if question:
            retriever = st.session_state.vectorstore.as_retriever()
            docs = retriever.get_relevant_documents(question)

            # optional reranking
            if use_reranker:
                scored = rerank(question, docs)
                docs = [d for _, d in scored]

            context = '\n\n'.join([d.page_content for d in docs[:5]])

            if mode == 'Q&A':
                prompt = PromptTemplate(
                    input_variables=['question', 'context'],
                    template=(
                        "You are a helpful medical assistant. Use the context to answer the question. "
                        "If unsure, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion:\n{question}"
                    ),
                )
            else:
                prompt = PromptTemplate(
                    input_variables=['instructions', 'context'],
                    template=(
                        "You are a concise summarization assistant for clinical documents. "
                        "Follow the user instructions and summarize the most important clinical points from the context.\n\n"
                        "Instructions:\n{instructions}\n\nContext:\n{context}"
                    ),
                )

            if make_llm is None:
                st.error('No LLM backend available. Install dependencies or set GROQ_API_KEY/OpenAI key.')
                st.stop()
            llm = make_llm(temperature=0)

            # build the final prompt text depending on mode
            if mode == 'Q&A':
                full_prompt = prompt.format(question=question, context=context)
            else:
                full_prompt = prompt.format(instructions=question, context=context)

            # resilient LLM call: prioritize .invoke for chat-style backends (Groq/langchain_groq),
            # then try direct call, then .generate. Extract text from common return shapes.
            def extract_text(result):
                # result might be a string
                if isinstance(result, str):
                    return result
                # dict-like
                if isinstance(result, dict):
                    # common key names
                    for k in ('content', 'text', 'response'):
                        if k in result and isinstance(result[k], str):
                            return result[k]
                    # if nested
                    if 'output' in result and isinstance(result['output'], str):
                        return result['output']
                    return str(result)
                # object with attributes
                if hasattr(result, 'content'):
                    return getattr(result, 'content')
                if hasattr(result, 'text'):
                    return getattr(result, 'text')
                # LangChain generation objects
                gens = getattr(result, 'generations', None)
                if gens:
                    try:
                        return gens[0][0].text
                    except Exception:
                        return str(gens)
                # fallback
                return str(result)

            def call_llm(llm_obj, text: str):
                # If the LLM supports invoke (chatty backends), prefer that
                try:
                    if hasattr(llm_obj, 'invoke'):
                        # many chat backends (Groq via langchain_groq) expect a PromptValue or str
                        out = llm_obj.invoke(text)
                        return extract_text(out)
                except Exception as e:
                    # continue to other methods but log
                    st.write(f'LLM.invoke failed: {e}')

                # try direct call
                try:
                    out = llm_obj(text)
                    return extract_text(out)
                except Exception as e:
                    st.write(f'LLM direct call failed: {e}')

                # try generate
                try:
                    if hasattr(llm_obj, 'generate'):
                        out = llm_obj.generate([text])
                        return extract_text(out)
                except Exception as e:
                    st.write(f'LLM.generate failed: {e}')

                return 'LLM call failed: unknown error or unsupported LLM interface'

            # clean context: normalize whitespace to avoid glued words
            import re
            def clean_text(s: str) -> str:
                s = re.sub(r"\s+", " ", s)
                # ensure a space after punctuation if missing (basic heuristic)
                s = re.sub(r'([\.,;:\?!])([A-Za-z0-9])', r'\1 \2', s)
                return s.strip()

            # apply cleaning to context
            context = '\n\n'.join([clean_text(d.page_content) for d in docs[:5]])

            response = call_llm(llm, full_prompt)
            st.subheader('Answer')
            st.write(response)
            with st.expander('Retrieved documents'):
                for i, d in enumerate(docs[:5], 1):
                    st.markdown(f'**Doc {i} — source:** {d.metadata.get("source")}')
                    st.write(d.page_content[:1000])
        else:
            st.warning('Please enter a question')
