"""Structured PDF parsing using Docling with PyMuPDF fallback."""

from pathlib import Path

from kisan.core.logging import logger


def parse_pdf_to_markdown(pdf_path: str | Path) -> str:
    """Parse a PDF file into structured Markdown text.

    Uses Docling for structure-aware extraction (headings, tables, lists).
    Falls back to PyMuPDF plain text extraction if Docling fails.

    Returns:
        Markdown string with the full document content.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        return _parse_with_docling(pdf_path)
    except Exception as e:
        logger.warning(f"Docling parsing failed for {pdf_path.name}, falling back to PyMuPDF: {e}")
        return _parse_with_pymupdf(pdf_path)


def _parse_with_docling(pdf_path: Path) -> str:
    """Parse PDF using Docling for structured Markdown output."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()

    if not markdown.strip():
        raise ValueError("Docling produced empty output")

    logger.info(f"Parsed {pdf_path.name} with Docling ({len(markdown)} chars)")
    return markdown


def _parse_with_pymupdf(pdf_path: Path) -> str:
    """Fallback: parse PDF using PyMuPDF plain text extraction."""
    import fitz

    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append(f"<!-- page {page_num} -->\n{text}")
    doc.close()

    full_text = "\n\n".join(pages)
    logger.info(f"Parsed {pdf_path.name} with PyMuPDF fallback ({len(full_text)} chars)")
    return full_text
