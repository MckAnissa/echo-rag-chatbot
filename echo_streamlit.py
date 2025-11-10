#!/usr/bin/env python3
"""
Echo UI - Streamlit interface for the Echo chatbot
"""

import streamlit as st
from echo_rag import EchoChatbot, knowledge_base
import time

# Page config
st.set_page_config(
    page_title="Echo - Your Introspective AI Companion",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
    }
    div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
    }
    .user-message {
        background-color: #1e3a5f;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .bot-message {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot" not in st.session_state:
    st.session_state.bot = None

if "bot_initialized" not in st.session_state:
    st.session_state.bot_initialized = False

# Initialize bot - FIXED VERSION (no caching initially)
def initialize_bot():
    """Initialize Echo chatbot"""
    bot = EchoChatbot(model_name="microsoft/phi-2")
    bot.load_knowledge_base(knowledge_base)
    return bot

# Header
st.title("🧠 Echo - Your Introspective AI Companion")
st.markdown("""
A locally-run RAG chatbot that thinks deeply about ethics, philosophy, technology, and human experience.
Echo has access to a curated knowledge base and engages in thoughtful, introspective conversations.
""")

# Load bot with progress indicator - FIXED
if not st.session_state.bot_initialized:
    with st.spinner("🔄 Initializing Echo... This may take 2-5 minutes on first run..."):
        try:
            st.session_state.bot = initialize_bot()
            st.session_state.bot_initialized = True
            st.success("✅ Echo is ready!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to initialize Echo: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About Echo")
    st.markdown("""
    Echo is a philosophical AI assistant running entirely on your local machine.
    
    **Features:**
    - 🔒 Completely private
    - 🧠 Introspective reasoning
    - 📚 Curated knowledge base
    - 💭 Deep conversations
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.bot:
            st.session_state.bot.reset_conversation()
        st.rerun()
    
    st.divider()
    
    st.markdown("### 💡 Example Questions")
    st.markdown("""
    - What are your thoughts on AI consciousness?
    - How should we balance privacy and security?
    - Can you help me think through a moral dilemma?
    - What's the difference between happiness and meaning?
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask Echo anything about ethics, philosophy, technology, or life..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant", avatar="🧠"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Echo is thinking... (this may take 30-120 seconds)")
        
        try:
            response = st.session_state.bot.chat(prompt)
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            import traceback
            st.code(traceback.format_exc())

# Footer
st.divider()
st.caption("⚡ Running locally with Phi-2 | 🔒 Your conversations are private")