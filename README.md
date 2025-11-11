# Echo — Local RAG Chatbot

Echo is a locally-run Retrieval-Augmented Generation (RAG) chatbot with a Streamlit web interface. It uses TF-IDF for document retrieval and runs Phi-2 (or other small language models) for generation, optimized for CPU-only environments.

## Features

- 🔒 **Privacy-first**: Runs completely locally on your machine
- 📚 **RAG capabilities**: Upload custom documents for context-aware responses
- 🧠 **Introspective AI**: Engages in thoughtful conversations on ethics, philosophy, and more
- 🖥️ **CPU optimized**: Works without a GPU (though slower)
- 🎨 **Modern UI**: Clean Streamlit interface with chat history
- ⚡ **Flexible deployment**: Run locally or with a separate model server

## Quick Start

### 1. Create and activate a Python virtual environment

**Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install torch transformers scikit-learn streamlit requests
```

**Optional (for 4-bit quantization on GPU):**
```bash
pip install bitsandbytes
```

### 3. Run the Streamlit app
```bash
streamlit run echo_streamlit.py
```

The app will open in your browser at `http://localhost:8501`

### 4. Initialize Echo

- In the sidebar, select your preferred model (default: `microsoft/phi-2`)
- Click "🚀 Initialize Echo (local)"
- Wait for the model to download and load (first run takes 5-10 minutes)
- Start chatting!

## Usage Tips

### For CPU Users (No GPU)

- **Expect slower responses**: 1-2 minutes per response on CPU is normal
- **Reduce max tokens**: Lower the "Max Response Length" slider to 50-100 tokens for faster responses
- **Try TinyLlama**: Switch to `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for 3x faster inference
- **Use remote server mode**: Keeps the UI responsive during long generations

### Adding Custom Knowledge

1. Click "Add Documents" in the sidebar
2. Paste your documents (one per line)
3. Click "Add to Knowledge Base"
4. Enable "Use Document Retrieval (RAG)" when chatting

### Remote Server Mode

For better performance with large models:

1. Select "Remote model server" mode in sidebar
2. Click "Start model server"
3. Click "Load Model on Server"
4. Wait for loading to complete
5. Chat with a responsive UI

## Project Structure
```
rag-echo/
├── echo_rag.py           # Core chatbot logic (EchoChatbot class)
├── echo_streamlit.py     # Streamlit web interface
├── model_server.py       # Optional FastAPI server for remote inference
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Requirements

- Python 3.8+
- 8GB+ RAM (16GB recommended for Phi-2)
- ~5GB disk space for model weights
- Internet connection (first run only, to download models)

## Models Supported

- **microsoft/phi-2** (default) - 2.7B parameters, good quality
- **microsoft/phi-3-mini-4k-instruct** - 3.8B parameters, slightly larger
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** - 1.1B parameters, fastest on CPU

## Built-in Knowledge Base

Echo comes with a curated knowledge base covering:
- Ethics and moral philosophy
- Human rights and political systems
- Animal welfare
- Technology and AI ethics
- Personal identity and consciousness
- And much more...

## Troubleshooting

### Import errors
```bash
# Make sure you're in the virtual environment
python -c "from echo_rag import EchoChatbot; print('Import successful')"
```

### Port conflicts
If Streamlit can't start on port 8501, it will automatically try 8502, 8503, etc.

### Model download issues
If downloads fail or corrupt:
```bash
# Clear the Hugging Face cache and retry
rm -rf ~/.cache/huggingface/hub/models--microsoft--phi-2
```

**Windows:**
```powershell
Remove-Item -Recurse -Force ~\.cache\huggingface\hub\models--microsoft--phi-2
```

### Memory issues
- Try TinyLlama instead of Phi-2
- Close other applications
- Reduce max_tokens in generation settings

## Contributing

Feel free to open issues or submit pull requests!

## License

[Your license here]

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Models from [Hugging Face](https://huggingface.co/)
- Inspired by the need for private, local AI assistants