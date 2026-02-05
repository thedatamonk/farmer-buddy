"""Streamlit UI for Kisan Mitra chatbot."""

import httpx
import streamlit as st

from components.chat import (
    display_chat_history,
    display_chat_message,
    display_processing_time,
    display_tools_used,
    display_welcome_message,
    get_chat_input,
)
from components.image_upload import display_uploaded_image, get_image_upload

# Page configuration
st.set_page_config(
    page_title="Kisan Mitra - Agricultural Assistant",
    page_icon="🌾",
    layout="wide",
)

# API endpoint
API_URL = "http://localhost:8080/api/v1"


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "image_base64" not in st.session_state:
        st.session_state.image_base64 = None
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None


def send_message(message: str, image_base64: str | None = None) -> dict | None:
    """Send message to the API and return response."""
    try:
        payload = {
            "message": message,
            "session_id": st.session_state.session_id,
        }
        if image_base64:
            payload["image"] = image_base64

        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{API_URL}/chat", json=payload)
            response.raise_for_status()
            return response.json()

    except httpx.ConnectError:
        st.error("Could not connect to the server. Make sure the API is running.")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API error: {e.response.status_code}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def main():
    """Main application."""
    init_session_state()

    # Header
    st.title("🌾 Kisan Mitra")
    st.caption("Your Agricultural Assistant | आपका कृषि सहायक")

    # Sidebar
    with st.sidebar:
        st.header("📸 Image Upload")
        image_base64, uploaded_image = get_image_upload()

        if image_base64:
            st.session_state.image_base64 = image_base64
            st.session_state.uploaded_image = uploaded_image
            display_uploaded_image(uploaded_image)
            st.success("Image ready for analysis!")

        st.divider()

        st.header("ℹ️ Quick Help")
        st.markdown("""
        **Disease Detection**
        Upload a crop photo and describe the problem.

        **Mandi Prices**
        Ask: "गेहूं का भाव क्या है?"

        **Government Schemes**
        Ask: "PM-KISAN क्या है?"
        """)

        st.divider()

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.session_state.image_base64 = None
            st.session_state.uploaded_image = None
            st.rerun()

    # Main chat area
    if not st.session_state.messages:
        display_welcome_message()

    # Display chat history
    display_chat_history(st.session_state.messages)

    # Chat input
    if prompt := get_chat_input():
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)

        # Get image if uploaded
        current_image = st.session_state.get("image_base64")

        # Show thinking indicator
        with st.spinner("Thinking..."):
            response = send_message(prompt, current_image)

        if response:
            # Update session ID
            st.session_state.session_id = response.get("session_id")

            # Add assistant response to history
            assistant_message = response.get("response", "")
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_message,
            })

            # Display response
            display_chat_message("assistant", assistant_message)

            # Display metadata
            col1, col2 = st.columns([3, 1])
            with col1:
                display_tools_used(response.get("tools_used", []))
            with col2:
                display_processing_time(response.get("processing_time_ms"))

            # Clear image after use
            if current_image:
                st.session_state.image_base64 = None
                st.session_state.uploaded_image = None


if __name__ == "__main__":
    main()
