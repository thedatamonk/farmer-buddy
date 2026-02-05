"""Chat interface component for Streamlit."""

import streamlit as st


def display_chat_message(role: str, content: str, avatar: str | None = None):
    """Display a chat message with appropriate styling."""
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🌾"):
            st.markdown(content)


def display_chat_history(messages: list[dict]):
    """Display the chat history."""
    for msg in messages:
        display_chat_message(msg["role"], msg["content"])


def get_chat_input() -> str | None:
    """Get user input from chat interface."""
    return st.chat_input("Ask about crops, diseases, prices, or government schemes...")


def display_tools_used(tools: list[dict]):
    """Display tools that were used in the response."""
    if not tools:
        return

    with st.expander("🔧 Tools Used", expanded=False):
        for tool in tools:
            st.markdown(f"**{tool['name']}**")
            if tool.get("input"):
                st.json(tool["input"])


def display_processing_time(time_ms: float | None):
    """Display processing time."""
    if time_ms:
        st.caption(f"⏱️ Response time: {time_ms:.0f}ms")


def display_welcome_message():
    """Display welcome message for new users."""
    st.markdown("""
    ### 🌾 Welcome to Kisan Mitra!

    I'm here to help Indian farmers with:

    - **🔬 Crop Disease Detection** - Upload a photo of your crop to identify diseases
    - **💰 Mandi Prices** - Get current market prices for your crops
    - **📋 Government Schemes** - Learn about subsidies and support programs

    **How to use:**
    1. Type your question in the chat below
    2. For disease detection, use the upload button to add a crop image
    3. Ask in Hindi or English - I understand both!

    **Example questions:**
    - "मेरी गेहूं की फसल में पीले धब्बे हैं" (My wheat crop has yellow spots)
    - "What is the price of onion in Maharashtra?"
    - "Tell me about PM-KISAN scheme"
    """)
