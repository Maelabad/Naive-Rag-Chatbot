import os
import sys
from typing import List
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils import path_to_loader


def ingest_paths(paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200, persist: bool = False, index_path: str = 'faiss_index'):
    docs = []
    for p in paths:
        loaded = path_to_loader(p)
        docs.extend(loaded)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    # choose embeddings backend: prefer HuggingFace local model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # persist/load
    if persist and os.path.exists(index_path):
        print('Loading existing FAISS index from', index_path)
        # Loading uses pickle under the hood; only enable if you trust the files
        vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        # Add newly ingested chunks to the existing index
        try:
            texts = [d.page_content for d in chunks]
            metadatas = [d.metadata for d in chunks]
            # FAISS vectorstore exposes add_texts
            vs.add_texts(texts, metadatas=metadatas)
            # save updated index
            vs.save_local(index_path)
        except Exception as e:
            print('Failed to add documents to existing FAISS index:', e)
        return vs

    vectorstore = FAISS.from_documents(chunks, embeddings)

    if persist:
        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)

    return vectorstore


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Ingest documents and build FAISS index')
    parser.add_argument('paths', nargs='+', help='Files to ingest')
    parser.add_argument('--out', default='faiss_index', help='Output index folder (ignored for in-memory)')
    args = parser.parse_args()

    vs = ingest_paths(args.paths)
    print('Indexed chunks:', len(vs.docstore._dict))
