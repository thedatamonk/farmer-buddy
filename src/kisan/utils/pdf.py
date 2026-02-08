"""PDF text extraction utilities."""

from pathlib import Path

import fitz  # PyMuPDF

from kisan.core.logging import logger


def extract_text_from_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract text from a PDF file with page information.

    Returns:
        List of dicts with 'text', 'page_number', and 'source' keys
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():  # Only include non-empty pages
                pages.append({
                    "text": text,
                    "page_number": page_num,
                    "source": pdf_path.name,
                })
        doc.close()
        logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")

    except Exception as e:
        logger.error(f"Failed to extract PDF {pdf_path}: {e}")
        raise

    return pages


# TODO: Improve chunking strategy
def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to split
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        # Find the end of this chunk
        end = start + chunk_size

        # If not at the end, try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence boundaries first
            for boundary in [". ", ".\n", "? ", "! ", "\n\n"]:
                boundary_pos = text.rfind(boundary, start, end + 50)
                if boundary_pos > start + chunk_size // 2:
                    end = boundary_pos + len(boundary)
                    break
            else:
                # Fall back to word boundary
                space_pos = text.rfind(" ", start, end)
                if space_pos > start + chunk_size // 2:
                    end = space_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = end - overlap

    return chunks


def process_pdf_for_indexing(
    pdf_path: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Process a PDF file into chunks ready for indexing.

    Returns:
        List of dicts with 'content', 'source', 'page_number', and 'chunk_index' keys
    """
    pages = extract_text_from_pdf(pdf_path)
    chunks = []
    chunk_index = 0

    for page in pages:
        page_chunks = chunk_text(page["text"], chunk_size, chunk_overlap)
        for page_chunk in page_chunks:
            chunks.append({
                "content": page_chunk,
                "source": page["source"],
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    logger.info(f"Created {len(chunks)} chunks from {pdf_path}")
    return chunks
