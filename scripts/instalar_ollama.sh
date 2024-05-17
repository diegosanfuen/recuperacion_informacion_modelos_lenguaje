#!/usr/bin/env sh
curl -fsSL https://ollama.com/install.sh | sh
ollama serve & ollama pull llama3

pip install langchain
pip install langchain-core
pip install langchain-community

pip install faiss-gpu
