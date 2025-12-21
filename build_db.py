import os
import yaml
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

def build_database():   
    pdf_path = config["files"]["pdf_path"]
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Splitting Documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, 
        chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)
    
    # Embedding & Save Vector to DB
    embeddings = OllamaEmbeddings(model=config["embeddings"]["model_name"])
    db_path = config["files"]["chroma_db_path"]
    
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="fraud_rag_db",
        embedding=embeddings,
        persist_directory=db_path
    )

if __name__ == "__main__":
    build_database()