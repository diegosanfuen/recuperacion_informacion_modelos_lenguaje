#!/usr/bin/env sh
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve &
ollama pull llama3
