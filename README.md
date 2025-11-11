# 🧠 Echo — Your Introspective AI Companion

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-FF4B4B.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-EE4C2C.svg)
![Maintenance](https://img.shields.io/badge/maintained-yes-brightgreen.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

> **A personal ML engineering project demonstrating RAG implementation, local LLM deployment, and full-stack AI application development.**

Echo is a privacy-first conversational AI that runs entirely on your local machine. No cloud services, no API costs, no data collection—just a thoughtful AI companion powered by Phi-2 and retrieval-augmented generation.

**What makes this project interesting:**
- Implements RAG from scratch using TF-IDF retrieval (no vector DB dependencies)
- Optimizes 2.7B parameter model (Phi-2) to run on CPU without GPU
- Builds production-ready UI with Streamlit including session management and caching
- Includes FastAPI model server option for async inference
- Demonstrates software engineering best practices (error handling, retry logic, modular design)

---

## ✨ Features

- 🔒 **Completely Local**: All processing happens on your machine—your conversations stay private
- 📚 **RAG-Powered**: Upload custom documents and Echo retrieves relevant context for informed responses
- 🧠 **Introspective Dialogue**: Engages thoughtfully on topics like ethics, philosophy, technology, and human experience
- 🖥️ **CPU Optimized**: Works without a GPU (though it's slower)
- 🎨 **Modern UI**: Clean, intuitive Streamlit interface with persistent chat history
- ⚡ **Flexible Deployment**: Run locally in-process or use a separate model server for better responsiveness
- 🎯 **Beginner-Friendly**: Clear setup instructions and helpful error messages

---

## 🎯 Technical Highlights

### Skills Demonstrated
- **RAG Implementation**: Built custom retrieval system using TF-IDF and cosine similarity for document-grounded responses
- **LLM Integration**: Implemented Hugging Face Transformers with Phi-2/TinyLlama models and CPU-optimized inference
- **UI/UX Design**: Created intuitive Streamlit interface with session state management and real-time chat updates
- **Error Handling**: Robust retry logic with exponential backoff for model loading and generation failures
- **Performance Optimization**: Memory-efficient caching using `@st.cache_resource` to prevent model reloading
- **Model Management**: Automatic device detection, quantization support, and tokenizer configuration

### Challenges Overcome
- **CPU Inference Optimization**: Implemented strategies to make Phi-2 (2.7B parameters) usable without GPU through careful memory management
- **Memory Management**: Used Streamlit's caching system to keep 5GB+ models in memory across reruns
- **Type Safety**: Resolved complex type hint issues with conditional imports and `TYPE_CHECKING` guards
- **Async UI**: Built responsive interface that doesn't freeze during long CPU-based generation cycles

---

## 📸 Screenshots

### Chat Interface
*Coming soon - will show the main conversation interface with Echo responding to questions*

### Knowledge Base Management
*Coming soon - will demonstrate the custom document upload and RAG functionality*

---

## 🚀 Quick Start

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
pip install -r requirements.txt
```

**Or install manually:**
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

---

## 💡 Usage Tips

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

---

## 📁 Project Structure

```
rag-echo/
├── echo_rag.py           # Core chatbot logic (EchoChatbot class, DocumentRetriever)
├── echo_streamlit.py     # Streamlit web interface with session management
├── model_server.py       # Optional FastAPI server for remote inference
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

**Key Components:**

- **`EchoChatbot`**: Main class handling model loading, generation, and conversation management
- **`DocumentRetriever`**: TF-IDF-based retrieval system for RAG functionality
- **`echo_streamlit.py`**: Streamlit UI with caching, error handling, and dual-mode support
- **`model_server.py`**: FastAPI backend for separating model inference from UI

---

## 🛠️ Requirements

- Python 3.8+
- 8GB+ RAM (16GB recommended for Phi-2)
- ~5GB disk space for model weights
- Internet connection (first run only, to download models)

---

## 🤖 Models Supported

- **microsoft/phi-2** (default) - 2.7B parameters, good quality, ~5GB
- **microsoft/phi-3-mini-4k-instruct** - 3.8B parameters, slightly larger
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** - 1.1B parameters, fastest on CPU

---

## 📚 Built-in Knowledge Base

Echo comes with a curated knowledge base covering:
- Ethics and moral philosophy
- Human rights and political systems
- Animal welfare and rights
- Technology and AI ethics
- Personal identity and consciousness
- Religion and spirituality
- Environmental issues
- And much more...

---

## 🔧 Troubleshooting

### Import errors
```bash
# Make sure you're in the virtual environment
python -c "from echo_rag import EchoChatbot; print('Import successful')"
```

### Port conflicts
If Streamlit can't start on port 8501, it will automatically try 8502, 8503, etc.

### Model download issues
If downloads fail or become corrupted:

**Linux/Mac:**
```bash
rm -rf ~/.cache/huggingface/hub/models--microsoft--phi-2
```

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force ~\.cache\huggingface\hub\models--microsoft--phi-2
```

Then restart the application.

### Memory issues
- Try TinyLlama instead of Phi-2
- Close other applications to free up RAM
- Reduce `max_tokens` in generation settings
- Use remote server mode to isolate model memory

### Slow response times
- **Normal on CPU**: 1-2 minutes per response is expected
- Lower max tokens (50-100 instead of 200)
- Switch to TinyLlama for 3x speed improvement
- Consider running on a machine with a GPU for real-time responses

---

## 🚀 Future Improvements

- [ ] Add embeddings-based retrieval (FAISS/ChromaDB) for better semantic search
- [ ] Implement conversation memory persistence across sessions
- [ ] Add GPU support with automatic CUDA detection and optimization
- [ ] Create Docker container for one-command deployment
- [ ] Add comprehensive unit tests and CI/CD pipeline
- [ ] Implement streaming responses for better UX (token-by-token display)
- [ ] Add support for PDF/DOCX document uploads
- [ ] Create REST API wrapper for programmatic access
- [ ] Add conversation export functionality (JSON/markdown)
- [ ] Implement multi-turn conversation context optimization
- [ ] Add support for fine-tuning on custom datasets
- [ ] Create web-based admin panel for knowledge base management

---

## 🤝 Contributing

This is a personal learning project, but contributions, issues, and feature requests are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Models from [Hugging Face](https://huggingface.co/) (Phi-2, TinyLlama)
- Inspired by the need for private, local AI assistants
- Thanks to the open-source ML community for excellent tools and documentation

---

## 👤 About

Built by Anissa as a personal learning project to explore RAG systems, local LLM deployment, and conversational AI.

This project demonstrates practical skills in:
- Machine Learning Engineering
- Natural Language Processing
- Full-Stack Development
- Software Engineering Best Practices

**Send questions to:**
- Email: [MckAnissa@proton.me]
- GitHub: [@MckAnissa](https://github.com/MckAnissa)

---

<div align="center">
  
**⭐ If you found this project interesting, consider giving it a star!**

Built with ❤️ and lots of ☕

</div>