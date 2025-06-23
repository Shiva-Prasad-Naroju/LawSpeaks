import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load the pdfs
# We have 4 pdf so combining them altogether.
pdfs_folder = "data"
all_docs = []

for file in os.listdir(pdfs_folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdfs_folder, file))
        docs = loader.load()
        all_docs.extend(docs)

print(f"✅ Loaded {len(all_docs)} documents.")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=80) # tried with diff values
# splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64) # try this, may imrove sematic search


chunked_docs = splitter.split_documents(all_docs)

print(f"✅ Split into {len(chunked_docs)} chunks.")

# Step 3: Embed and store in vector DB
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")


df_faiss = FAISS.from_documents(chunked_docs, embeddings)
df_faiss.save_local("vector_db_1")

print("✅ Vector_db_1 created and saved.")
