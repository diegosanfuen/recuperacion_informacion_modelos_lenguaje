#!/usr/bin/env sh
curl -fsSL https://ollama.com/install.sh | sh
ollama serve & ollama pull llama3

