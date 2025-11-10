# Echo — Local RAG Chatbot

Echo is a local Retrieval-Augmented Generation (RAG) chatbot. It uses sentence-transformers for embeddings and a local Hugging Face-compatible causal language model for generation. The project includes a small built-in knowledge base and a simple CLI chat loop.

## Quick start

1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install numpy torch sentence-transformers transformers bitsandbytes
