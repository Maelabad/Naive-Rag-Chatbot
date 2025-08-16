import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, CSVLoader
import pdfplumber
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def load_text_file(path: str) -> List[Document]:
    loader = TextLoader(path, encoding='utf-8')
    return loader.load()


def load_csv_file(path: str) -> List[Document]:
    loader = CSVLoader(path)
    return loader.load()


def load_pdf_file(path: str) -> List[Document]:
    docs = []
    with pdfplumber.open(path) as pdf:
        text = []
        for page in pdf.pages:
            text.append(page.extract_text() or "")
        content = "\n\n".join(text)
        docs.append(Document(page_content=content, metadata={"source": path}))
    return docs


def path_to_loader(path: str) -> List[Document]:
    # handle URLs
    parsed = urlparse(path)
    if parsed.scheme in ('http', 'https'):
        return load_url(path)

    _, ext = os.path.splitext(path.lower())
    if ext in ('.txt', '.md'):
        return load_text_file(path)
    elif ext in ('.csv',):
        return load_csv_file(path)
    elif ext in ('.pdf',):
        return load_pdf_file(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_url(url: str) -> List[Document]:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    # naive extraction: join visible text
    texts = []
    for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']):
        t = p.get_text(separator=' ', strip=True)
        if t:
            texts.append(t)
    content = '\n\n'.join(texts)
    return [Document(page_content=content, metadata={'source': url})]
