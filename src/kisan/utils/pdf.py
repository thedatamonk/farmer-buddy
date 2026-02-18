"""PDF processing utilities: parsing and structure-aware chunking."""

from pathlib import Path

from kisan.core.logging import logger
from kisan.utils.chunker import MarkdownChunker
from kisan.utils.pdf_parser import parse_pdf_to_markdown


def process_pdf_for_indexing(
    pdf_path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """Process a PDF file into enriched chunks ready for indexing.

    Uses Docling for structured PDF parsing, then a token-based
    Markdown-aware chunker with metadata extraction.

    Returns:
        List of dicts with keys: content, source, scheme_name,
        section_header, section_hierarchy, content_type,
        page_numbers, chunk_index
    """
    pdf_path = Path(pdf_path)
    markdown = parse_pdf_to_markdown(pdf_path)

    chunker = MarkdownChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = chunker.chunk(markdown, source=pdf_path.name)

    logger.info(f"Created {len(chunks)} chunks from {pdf_path.name}")
    return chunks
