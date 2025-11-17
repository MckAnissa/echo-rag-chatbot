import streamlit as st
import sys
from pathlib import Path

# Import the bot from echo_rag
from echo_rag import EchoChatbot, knowledge_base

st.set_page_config(
    page_title="Echo - Local RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_bot(model_path: str, n_threads: int = 4, n_ctx: int = 2048):
    """
    Load the bot with caching to prevent reloading on every interaction.
    This is critical for performance.
    """
    try:
        bot = EchoChatbot(
            model_path=model_path,
            load_model=True,
            n_ctx=n_ctx,
            n_threads=n_threads
        )
        bot.load_knowledge_base(knowledge_base)
        return bot, None
    except Exception as e:
        return None, str(e)

# Sidebar configuration
with st.sidebar:
    st.title("Echo Configuration")

    # Model selection
    model_path = st.text_input(
        "Model Path",
        value="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        help="Path to GGUF model file"
    )

    # Thread configuration
    n_threads = st.slider("CPU Threads", 1, 8, 4)
    n_ctx = st.slider("Context Window", 512, 4096, 2048, step=512)

    # Load button
    if st.button("Load/Reload Model", type="primary"):
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # Generation settings
    st.subheader("Generation Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max Response Length", 50, 300, 128, 10)

    st.divider()

    # Clear conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        if "bot" in st.session_state and st.session_state.bot:
            try:
                st.session_state.bot.reset_conversation()
            except Exception:
                pass
        st.rerun()

    # Quick debug button to inspect what bot.chat returns
    if st.button("DEBUG: inspect bot.chat return"):
        st.session_state._debug_inspect = True

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot" not in st.session_state:
    st.session_state.bot = None
    st.session_state.error = None

if st.session_state.bot is None:
    with st.spinner("Loading model... This may take a minute on first run."):
        # Use the FULL PATH to the model (change if necessary)
        model_full_path = r"C:\Users\lilit\Projects\rag-echo-v2\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        bot, error = load_bot(model_full_path, 4, 4096)
        st.session_state.bot = bot
        st.session_state.error = error

# Show error if model failed to load
if st.session_state.error:
    st.error(f"Failed to load model: {st.session_state.error}")
    st.info("Please check the model path and try reloading.")
    st.stop()

# Show success message
if st.session_state.bot:
    st.success("Model loaded successfully")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------
# Chat input
# ---------------------------

import concurrent.futures
import inspect
import asyncio
from typing import Tuple

# Ensure a single executor across reruns
if "_EXECUTOR" not in st.session_state:
    st.session_state._EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def _call_bot_in_thread(bot, prompt: str, timeout: int = 180) -> Tuple[bool, str]:
    """
    Run bot.chat(prompt) in a worker thread and normalize the result to a single string.
    Returns (ok, result_or_error).
    Handles:
      - blocking functions returning str
      - generator yielding chunks (iterates fully and joins)
      - async coroutines (runs in a fresh event loop in the worker thread)
    """
    def worker():
        res = bot.chat(prompt)

        # If it's awaitable/coroutine, run it in a new loop for this thread
        if inspect.isawaitable(res):
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                final = loop.run_until_complete(res)
                return final if isinstance(final, str) else str(final)
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        # If it's an iterable/generator (but not a string/bytes), iterate fully
        if hasattr(res, "__iter__") and not isinstance(res, (str, bytes)):
            collected = ""
            for chunk in res:
                if chunk is None:
                    continue
                collected += str(chunk)
            return collected

        # Fallback: convert to string
        return str(res)

    future = st.session_state._EXECUTOR.submit(worker)
    try:
        out = future.result(timeout=timeout)
        return True, out
    except Exception as e:
        try:
            future.cancel()
        except Exception:
            pass
        return False, str(e)

# Session flags to avoid double-processing across Streamlit reruns
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# Capture user input (this sets pending_prompt and survives reruns)
user_input = st.chat_input("Ask Echo anything...")
if user_input:
    # Append user message once (avoid duplicates by checking last message)
    last = st.session_state.messages[-1] if st.session_state.messages else None
    if not (last and last["role"] == "user" and last["content"] == user_input):
        st.session_state.messages.append({"role": "user", "content": user_input})

    st.session_state.pending_prompt = user_input
    # Let rerun occur; the processing block below will handle the work

# If there's a pending prompt and we're not already processing it, handle it
if st.session_state.pending_prompt and not st.session_state.processing:
    st.session_state.processing = True
    prompt = st.session_state.pending_prompt

    # Render the user message immediately (so UI shows it)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant reply in worker thread
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if not st.session_state.bot:
                err = "Bot not loaded"
                st.error(err)
                last2 = st.session_state.messages[-1] if st.session_state.messages else None
                if not (last2 and last2["role"] == "assistant" and last2["content"] == err):
                    st.session_state.messages.append({"role": "assistant", "content": err})
            else:
                # Safely set generation kwargs if available
                if hasattr(st.session_state.bot, "generation_kwargs"):
                    try:
                        st.session_state.bot.generation_kwargs["temperature"] = temperature
                        st.session_state.bot.generation_kwargs["max_tokens"] = max_tokens
                    except Exception:
                        # ignore if bot doesn't expose that dict
                        pass

                ok, result = _call_bot_in_thread(st.session_state.bot, prompt, timeout=180)

                # Optional debug inspect button behavior (shows repr of bot.chat return in sidebar)
                if "_debug_inspect" in st.session_state and st.session_state._debug_inspect:
                    try:
                        val = st.session_state.bot.chat("diagnostic ping")
                        st.sidebar.write("repr:", repr(val))
                        # if generator, show first few chunks (non-destructive)
                        if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                            for i, chunk in enumerate(val):
                                st.sidebar.write(i, chunk)
                                if i >= 2:
                                    break
                    except Exception as e:
                        st.sidebar.exception(e)
                    finally:
                        st.session_state._debug_inspect = False

                if ok:
                    # Prevent duplicated assistant message
                    last2 = st.session_state.messages[-1] if st.session_state.messages else None
                    if not (last2 and last2["role"] == "assistant" and last2["content"] == result):
                        st.session_state.messages.append({"role": "assistant", "content": result})
                    st.markdown(result)
                else:
                    err = f"Error generating response: {result}"
                    st.error(err)
                    last2 = st.session_state.messages[-1] if st.session_state.messages else None
                    if not (last2 and last2["role"] == "assistant" and last2["content"] == err):
                        st.session_state.messages.append({"role": "assistant", "content": err})

    # Clear pending flags so app can accept next input
    st.session_state.pending_prompt = None
    st.session_state.processing = False

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"Messages: {len(st.session_state.messages)}")
with col2:
    if st.session_state.bot:
        tokens_estimate = sum(len(m["content"]) // 4 for m in st.session_state.messages)
        st.caption(f"Est. Tokens: {tokens_estimate}")
with col3:
    st.caption("CPU Mode")
