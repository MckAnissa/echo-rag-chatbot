# Echo - Local RAG Chatbot

A privacy-first conversational AI that runs entirely on your local machine using GGUF quantized models. No cloud services, no API costs, no data collection.

## Technical Highlights

This project demonstrates practical ML engineering skills:

- **RAG Implementation**: Custom retrieval system using TF-IDF and cosine similarity for document-grounded responses
- **CPU-Optimized Inference**: Uses llama-cpp-python for efficient CPU inference with GGUF quantized models
- **Modern UI**: Streamlit interface with proper caching, session state management, and real-time updates
- **Conversation Memory**: Persistent chat history with automatic compression to manage context windows
- **Error Handling**: Robust retry logic with exponential backoff for model loading and generation
- **Performance Optimization**: Memory-efficient caching prevents model reloading across sessions

## Features

- Completely local processing - your conversations stay private
- RAG-powered responses using custom knowledge base
- Persistent conversation memory with automatic summarization
- CPU-optimized with configurable threading
- Clean Streamlit web interface
- CLI mode for testing and automation

## Requirements

- Python 3.10+
- 8GB RAM minimum (16GB recommended)
- GGUF model file (Phi-2 Q4_K_M or similar)
- Windows, Linux, or macOS

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/MckAnissa/rag-echo-v2.git
cd rag-echo-v2
python -m venv venv
```

### 2. Activate virtual environment

Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```

Linux/Mac:
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download a GGUF model

Download Phi-2 Q4_K_M from HuggingFace:
https://huggingface.co/TheBloke/phi-2-GGUF

Place the .gguf file in the project root directory.

### 5. Run

CLI mode:
```bash
python echo_rag.py --model-path phi-2.Q4_K_M.gguf
```

Streamlit UI:
```bash
streamlit run echo_streamlit.py
```

## Configuration

### CLI Arguments

```bash
python echo_rag.py --model-path <path> --n-threads 4 --n-ctx 2048
```

- `--model-path`: Path to GGUF model file
- `--n-threads`: Number of CPU threads (default: 4)
- `--n-ctx`: Context window size (default: 2048)

### Setting Default Model Path

Edit line 618 in echo_rag.py:
```python
parser.add_argument("--model-path", type=str, default="phi-2.Q4_K_M.gguf", ...)
```

## Project Structure

```
rag-echo-v2/
├── echo_rag.py          # Core bot logic and CLI interface
├── echo_streamlit.py    # Streamlit web UI with caching
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── echo_memory.json    # Auto-generated conversation history
```

## How It Works

1. **Document Retrieval**: TF-IDF vectorization finds relevant knowledge base entries
2. **Context Building**: Combines retrieved documents with conversation history
3. **Generation**: llama-cpp-python performs efficient CPU inference on GGUF models
4. **Memory Management**: Automatically compresses old conversations into summaries

## Performance

Typical response times on modern CPU:
- Model loading: 5-10 seconds (first run only)
- Per message: 10-30 seconds depending on length
- Memory usage: 4-6GB RAM

## Built-in Knowledge Base

Echo includes curated knowledge on:
- Ethics and moral philosophy
- Human rights and political systems
- Animal welfare and rights
- Technology and AI ethics
- Personal identity and consciousness
- Religion and spirituality
- Environmental issues

## Troubleshooting

### Model not found

Verify the .gguf file path:
```bash
dir *.gguf  # Windows
ls *.gguf   # Linux/Mac
```

Use the exact filename (case-sensitive):
```bash
python echo_rag.py --model-path "phi-2.Q4_K_M.gguf"
```

### llama-cpp-python installation issues

On Windows, ensure Visual Studio Build Tools are installed.

Try upgrading:
```bash
pip install --upgrade llama-cpp-python
```

### Out of memory

- Reduce context window: `--n-ctx 1024`
- Use smaller quantization (Q4_0 instead of Q4_K_M)
- Close other applications
- Use a machine with more RAM

### Slow responses

This is normal on CPU. To improve:
- Reduce max_tokens in generation settings
- Use fewer CPU threads if system is overloaded
- Consider using a GPU-enabled machine for real-time responses

## Future Improvements

- Add embeddings-based retrieval (FAISS/ChromaDB)
- Implement streaming responses
- Add PDF/DOCX document upload
- GPU support with automatic CUDA detection
- Docker container for easy deployment
- REST API wrapper
- Conversation export functionality

## Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with Streamlit for the web interface
- Uses llama-cpp-python for efficient CPU inference
- Models from HuggingFace (Phi-2 and others)
- Inspired by the need for private, local AI assistants

## About

Built by Anissa McKnight as a personal learning project exploring RAG systems, local LLM deployment, and conversational AI.

**Contact:**
- Email: MckAnissa@proton.me
- GitHub: [@MckAnissa](https://github.com/MckAnissa)

---

If you found this project interesting, consider giving it a star.