#!/usr/bin/env sh
pip install langchain
pip install langchain-core
pip install langchain-community
pip install faiss-gpu
pip install python-dotenv
pip install gradio
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

