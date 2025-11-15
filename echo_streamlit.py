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
        value="phi-2.Q4_K_M.gguf",
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
            st.session_state.bot.reset_conversation()
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot" not in st.session_state:
    st.session_state.bot = None
    st.session_state.error = None

# Load bot on first run - use fixed params to ensure caching works
if st.session_state.bot is None:
    with st.spinner("Loading model... This may take a minute on first run."):
        # Use the FULL PATH to the model
        model_full_path = r"C:\Users\lilit\Projects\rag-echo-v2\phi-2.Q4_K_M.gguf"
        bot, error = load_bot(model_full_path, 4, 2048)
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

# Chat input
if prompt := st.chat_input("Ask Echo anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.bot:
                    st.session_state.bot.generation_kwargs["temperature"] = temperature
                    st.session_state.bot.generation_kwargs["max_tokens"] = max_tokens
                    
                    response = st.session_state.bot.chat(prompt)
                    st.markdown(response)

                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Bot is not loaded!")
                    
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

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