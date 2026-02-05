"""Image upload component for Streamlit."""

import base64
from io import BytesIO

import streamlit as st
from PIL import Image


def get_image_upload() -> tuple[str | None, Image.Image | None]:
    """Get uploaded image and return base64 encoded string.

    Returns:
        Tuple of (base64_string, PIL Image) or (None, None) if no image
    """
    uploaded_file = st.file_uploader(
        "Upload crop image for disease detection",
        type=["jpg", "jpeg", "png"],
        help="Take a clear photo of the affected part of your crop",
    )

    if uploaded_file is not None:
        # Read and display the image
        image = Image.open(uploaded_file)

        # Convert to RGB if necessary (handles PNG with transparency)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        # Resize if too large (max 2048px on longest side)
        max_size = 2048
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return base64_string, image

    return None, None


def display_uploaded_image(image: Image.Image | None):
    """Display the uploaded image in the UI."""
    if image:
        st.image(image, caption="Uploaded crop image", use_container_width=True)


def clear_image_upload():
    """Clear the uploaded image from session state."""
    if "uploaded_image" in st.session_state:
        del st.session_state["uploaded_image"]
    if "image_base64" in st.session_state:
        del st.session_state["image_base64"]
