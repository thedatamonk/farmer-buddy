"""Tests for PDF chunking utilities."""

from kisan.utils.chunker import MarkdownChunker


def test_chunk_empty():
    """Test chunking empty text."""
    chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
    result = chunker.chunk("")
    assert result == []


def test_chunk_small_text():
    """Test chunking text smaller than chunk size."""
    chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
    result = chunker.chunk("This is a small text.")
    assert len(result) == 1
    assert result[0]["content"] == "This is a small text."


def test_chunk_preserves_sections():
    """Test that section headers are extracted as metadata."""
    md = """# My Scheme

## Eligibility

Must be a farmer.

## Benefits

Rs 6000 per year.
"""
    chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(md)

    headers = [c["section_header"] for c in chunks]
    assert "Eligibility" in headers
    assert "Benefits" in headers


def test_chunk_table_kept_atomic():
    """Test that tables are not split across chunks."""
    table = """\
## Data

| Col A | Col B |
|-------|-------|
| 1     | 2     |
| 3     | 4     |
| 5     | 6     |
"""
    chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(table)

    # The table should appear in a single chunk
    table_chunks = [c for c in chunks if c["content_type"] == "table"]
    assert len(table_chunks) == 1
    assert "Col A" in table_chunks[0]["content"]
    assert "5" in table_chunks[0]["content"]


def test_chunk_extracts_scheme_name():
    """Test scheme name extraction from H1."""
    md = "# PM-KISAN Scheme\n\nSome description."
    chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(md)
    assert chunks[0]["scheme_name"] == "PM-KISAN Scheme"


def test_chunk_content_type_detection():
    """Test content type classification."""
    md = """\
## Info

- Item one
- Item two
- Item three
"""
    chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(md)
    list_chunks = [c for c in chunks if c["content_type"] == "list"]
    assert len(list_chunks) >= 1


def test_chunk_large_section_is_split():
    """Test that large sections are recursively split."""
    long_text = "## Section\n\n" + ("This is a sentence. " * 200)
    chunker = MarkdownChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(long_text)
    assert len(chunks) > 1
    for c in chunks:
        # Each chunk should be within a reasonable range of the target
        assert chunker.token_count(c["content"]) <= chunker.chunk_size * 1.5


def test_chunk_section_hierarchy():
    """Test that section hierarchy is built correctly."""
    md = """\
# Scheme

## Eligibility

### Land Ownership

Must own land.
"""
    chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(md)

    land_chunks = [c for c in chunks if c["section_header"] == "Land Ownership"]
    assert len(land_chunks) == 1
    assert "Eligibility" in land_chunks[0]["section_hierarchy"]
    assert "Land Ownership" in land_chunks[0]["section_hierarchy"]
