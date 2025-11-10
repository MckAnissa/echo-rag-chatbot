#!/usr/bin/env python3
"""
echo_streamlit.py

Streamlit UI for Echo (local RAG chatbot).

Features:
- Cacheable EchoChatbot instance to avoid reloading on every rerun (st.cache_resource).
- Fast UI testing mode that skips heavy model downloads (load_model=False).
- Optional integration with a separate model server (model_server.py) that exposes /status, /load, /generate.
- Basic generation settings and knowledge base upload.
- Shows Python executable used by Streamlit so you can verify venv.

Usage:
    streamlit run echo_streamlit.py
"""

from typing import Optional, Dict, Any
import sys
import os
import time
import socket
import subprocess
import requests

import streamlit as st

# Make sure we can import your EchoChatbot implementation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from echo_rag import EchoChatbot
except Exception as e:
    EchoChatbot = None
    import_error_for_echo = e
else:
    import_error_for_echo = None

# -----------------------------
# Configuration constants
# -----------------------------
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# -----------------------------
# Helper utilities
# -----------------------------
def is_port_open(host: str, port: int) -> bool:
    """Check if a TCP port is open (used to detect model server)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.35)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False

def start_model_server_process(python_exe: str = sys.executable) -> subprocess.Popen:
    """
    Start the FastAPI model server (model_server.py) using the current Python executable.
    Returns Popen instance.
    """
    uvicorn_cmd = [python_exe, "-m", "uvicorn", "model_server:app", "--host", SERVER_HOST, "--port", str(SERVER_PORT)]
    # Note: stdout/stderr are piped so Streamlit won't be spammed; logs still available if you inspect process
    popen = subprocess.Popen(uvicorn_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return popen

def trigger_server_load(model_name: str) -> Dict[str, Any]:
    try:
        r = requests.post(f"{SERVER_URL}/load", params={"model_name": model_name}, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def ask_server(prompt: str, use_retrieval: bool = True, timeout: int = 120) -> str:
    """
    Ask the remote model server to generate a reply.
    Returns the assistant text or an error message string.
    """
    try:
        r = requests.post(f"{SERVER_URL}/generate", json={"prompt": prompt, "use_retrieval": use_retrieval}, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("response", "")
        else:
            # Include server text for debugging
            return f"[Server error {r.status_code}] {r.text}"
    except Exception as e:
        return f"[Request failed] {e}"

def poll_server_status(timeout: int = 600, poll_interval: float = 1.0) -> Dict[str, Any]:
    """
    Poll /status until ready or timeout. Returns last status JSON (or a dict with error on failure).
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{SERVER_URL}/status", timeout=3)
            if r.status_code == 200:
                j = r.json()
                if j.get("ready"):
                    return j
                # else keep polling
            else:
                pass
        except Exception:
            pass
        time.sleep(poll_interval)
    return {"ready": False, "error": "timeout"}

# -----------------------------
# Cached factory for local chatbot
# -----------------------------
@st.cache_resource
def get_chatbot_local(model_name: str, load_model: bool = True) -> Optional[EchoChatbot]:
    """
    Returns an EchoChatbot instance cached by Streamlit.
    If load_model=False, an instance is created that skips heavy loading (EchoChatbot must support load_model flag).
    """
    if EchoChatbot is None:
        raise ImportError(f"EchoChatbot import failed: {import_error_for_echo}")
    # The EchoChatbot implementation we provided supports load_model flag. Use it.
    bot = EchoChatbot(model_name=model_name, load_model=load_model)
    return bot

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Echo - Your Introspective AI Companion", page_icon="🧠", layout="wide")

# Styling (kept from your original)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTextInput > div > div > input { background-color: #1e1e1e; }
    .chat-message { padding: 1.2rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; flex-direction: column; }
    .user-message { background-color: #2b313e; border-left: 4px solid #4a9eff; color: #fff; }
    .assistant-message { background-color: #1e1e1e; border-left: 4px solid #ff4a9e; color: #fff; }
    .message-header { font-weight: bold; margin-bottom: 0.5rem; color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

# Session state defaults
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_base_loaded" not in st.session_state:
    st.session_state.knowledge_base_loaded = False
if "_server_process" not in st.session_state:
    st.session_state._server_process = None
if "_server_started_by_ui" not in st.session_state:
    st.session_state._server_started_by_ui = False

# Sidebar
with st.sidebar:
    st.title("🧠 Echo Configuration")

    # show python executable to help debug venv issues
    st.caption("Python executable used by this Streamlit process:")
    st.text(sys.executable)

    model_name = st.selectbox(
        "Select Model",
        ["microsoft/phi-2", "microsoft/phi-3-mini-4k-instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
        index=0,
        help="Choose the language model to use"
    )

    st.divider()

    # Option: run local cached bot (fast UI) or use a separate server
    st.subheader("Initialization Mode")
    mode = st.radio("Mode", options=["Local (cached in Streamlit)", "Remote model server (recommended for heavy models)"], index=0)

    if mode == "Local (cached in Streamlit)":
        skip_model = st.checkbox("Skip heavy model load (fast UI testing)", value=False,
                                 help="If checked, Echo will be created without loading tokenizer/model (useful for UI/dev).")
        if not st.session_state.initialized:
            if st.button("🚀 Initialize Echo (local)", type="primary", use_container_width=True):
                # Attempt to create cached local instance
                try:
                    st.session_state.chatbot = get_chatbot_local(model_name=model_name, load_model=not skip_model)
                    st.session_state.initialized = True
                    st.success("Echo instance created.")
                    if skip_model:
                        st.info("Model load was skipped. Add docs and test retrieval. Click the 'Load Model' button in sidebar to load the actual model.")
                    else:
                        st.info("Model/tokenizer loaded (this may have taken a while).")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to initialize local Echo: {e}")
        else:
            st.success("✅ Echo local instance ready")
            if st.session_state.chatbot:
                dev = getattr(st.session_state.chatbot, "device", "unknown")
                st.info(f"🖥️ Running on: **{dev.upper()}**")

            # If created with skip_model=True, provide button to load model into the cached instance by clearing cache and creating new one
            if getattr(st.session_state.chatbot, "tokenizer", None) is None:
                if st.button("Load Model Now (local)", use_container_width=True):
                    # clear resource cache by calling get_chatbot_local with load_model=True — Streamlit cache keys by arguments so passing load_model True will create a new entry
                    try:
                        st.session_state.chatbot = get_chatbot_local(model_name=model_name, load_model=True)
                        st.success("Model loaded locally.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to load model locally: {e}")

    else:
        # Remote server mode
        st.markdown("Use a separate process to host the model. This keeps Streamlit responsive.")
        server_running = is_port_open(SERVER_HOST, SERVER_PORT)
        if server_running:
            st.success(f"Model server appears to be running at {SERVER_URL}")
        else:
            st.warning(f"Model server not detected at {SERVER_URL}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start model server (spawn uvicorn)", use_container_width=True):
                if is_port_open(SERVER_HOST, SERVER_PORT):
                    st.info("Server already running.")
                else:
                    try:
                        proc = start_model_server_process()
                        st.session_state._server_process = proc
                        st.session_state._server_started_by_ui = True
                        st.info("Model server starting in background. Wait a few seconds, then click 'Load Model on Server'.")
                    except Exception as e:
                        st.error(f"Failed to start server process: {e}")
        with col2:
            if st.button("Load Model on Server (background)", use_container_width=True):
                if not is_port_open(SERVER_HOST, SERVER_PORT):
                    st.error("Server is not running. Start it first (or start it manually with uvicorn).")
                else:
                    resp = trigger_server_load(model_name)
                    st.info(str(resp))

        # Status area: poll server status (light polling only)
        st.divider()
        try:
            r = requests.get(f"{SERVER_URL}/status", timeout=1.5)
            if r.status_code == 200:
                status = r.json()
                if status.get("ready"):
                    st.success(f"Server ready. Model: {status.get('model_name')}")
                    st.session_state.initialized = True
                    # fetch a proxy "chatbot" marker for the UI to use (server mode)
                    st.session_state.chatbot = {"server_mode": True}
                elif status.get("loading_started_at"):
                    st.info("Model loading in progress on server...")
                    if st.button("Poll until ready (blocks briefly)", use_container_width=True):
                        with st.spinner("Waiting for server to finish loading..."):
                            final = poll_server_status(timeout=1800, poll_interval=2.0)
                            if final.get("ready"):
                                st.success("Server model is ready now.")
                                st.session_state.initialized = True
                                st.session_state.chatbot = {"server_mode": True}
                                st.experimental_rerun()
                            else:
                                st.error("Polling timed out or failed.")
                else:
                    st.info("Server is up but model not loaded. Click 'Load Model on Server' to start loading.")
            else:
                st.warning("Unable to read server status.")
        except Exception:
            st.info("Server unreachable for status check (not running or blocked).")

    st.divider()

    # Knowledge base section (works with both local and server modes; local adds to in-memory retriever,
    # server would require additional endpoint to push documents — currently we only support local add).
    st.subheader("📚 Knowledge Base")
    with st.expander("Add Documents (local only)"):
        doc_input = st.text_area(
            "Enter documents (one per line)",
            height=150,
            placeholder="One document per line. Example:\nPython is a programming language.\nAI stands for Artificial Intelligence.",
        )
        if st.button("Add to Knowledge Base (local)", use_container_width=True):
            if not doc_input:
                st.warning("Please paste or type documents to add.")
            elif not st.session_state.initialized:
                st.warning("Initialize Echo first (local mode) before adding documents.")
            elif isinstance(st.session_state.chatbot, dict) and st.session_state.chatbot.get("server_mode"):
                st.warning("You're in remote server mode. This local 'add' won't affect the remote server.")
            else:
                docs = [d.strip() for d in doc_input.split("\n") if d.strip()]
                try:
                    # Use the local chatbot's retriever if present
                    bot = st.session_state.chatbot
                    if bot is None:
                        st.warning("No local chatbot instance found.")
                    else:
                        bot.load_knowledge_base(docs)
                        st.session_state.knowledge_base_loaded = True
                        st.success(f"Added {len(docs)} documents to the local knowledge base.")
                except Exception as e:
                    st.error(f"Failed to add documents: {e}")

    st.divider()

    st.subheader("⚙️ Generation Settings")
    use_retrieval = st.checkbox("Use Document Retrieval (RAG)", value=True)
    max_tokens = st.slider("Max Response Length (tokens)", min_value=50, max_value=1000, value=200, step=50)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1)

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chatbot and not isinstance(st.session_state.chatbot, dict):
            try:
                st.session_state.chatbot.reset_conversation()
            except Exception:
                pass
        st.experimental_rerun()

# -----------------------------
# Main content area
# -----------------------------
st.title("🧠 Echo - Your Introspective AI Companion")
st.caption("A locally-run RAG chatbot (use remote server mode for heavy models).")

if import_error_for_echo:
    st.error(f"EchoChatbot import failed: {import_error_for_echo}")
    st.info("Make sure echo_rag.py is present and importable from this directory and you are running Streamlit in the same venv.")
    st.stop()

if not st.session_state.initialized:
    st.info("👈 Please initialize Echo using the sidebar to begin chatting.")
    st.markdown("""
    ### Welcome to Echo!

    Echo is a **locally-run RAG (Retrieval-Augmented Generation) chatbot** that:
    - 🔒 Runs either locally (in-process) or in a separate model server process
    - 📚 Uses document retrieval for informed responses (when using the local retriever)
    - 🧠 Engages in thoughtful, introspective conversations
    - ⚡ Supports CPU and GPU depending on your EchoChatbot implementation
    """)
else:
    # Display chat messages
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header">👤 You</div>
                <div>{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-header">🧠 Echo</div>
                <div>{content}</div>
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Ask Echo anything...")

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message immediately
        st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header">👤 You</div>
                <div>{prompt}</div>
            </div>
            """, unsafe_allow_html=True)

        # Update generation settings on local chatbot if present
        if st.session_state.chatbot and not isinstance(st.session_state.chatbot, dict):
            try:
                st.session_state.chatbot.generation_kwargs["max_new_tokens"] = max_tokens
                st.session_state.chatbot.generation_kwargs["temperature"] = temperature
            except Exception:
                pass

        # Generate response (either ask local chatbot or remote server)
        with st.spinner("Echo is thinking..."):
            try:
                if isinstance(st.session_state.chatbot, dict) and st.session_state.chatbot.get("server_mode"):
                    # Use server
                    response = ask_server(prompt, use_retrieval=use_retrieval)
                else:
                    # Local chat (EchoChatbot.chat)
                    bot = st.session_state.chatbot
                    if bot is None:
                        response = "[No chatbot instance available — initialize first]"
                    else:
                        response = bot.chat(prompt, use_retrieval=use_retrieval)
                st.session_state.messages.append({"role": "assistant", "content": response})
                # Rerun so the UI updates (Streamlit shows appended messages on rerun)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")

# Footer
st.divider()
st.caption("Built with ❤️ using Streamlit | Echo - Local RAG Chatbot")
