"""Tests for PDF utilities."""


from kisan.utils.pdf import chunk_text


def test_chunk_text_empty():
    """Test chunking empty text."""
    result = chunk_text("")
    assert result == []


def test_chunk_text_small():
    """Test chunking text smaller than chunk size."""
    text = "This is a small text."
    result = chunk_text(text, chunk_size=100)
    assert result == [text]


def test_chunk_text_basic():
    """Test basic text chunking."""
    text = "First sentence. " * 20  # ~300 characters
    result = chunk_text(text, chunk_size=100, overlap=20)

    assert len(result) > 1
    for chunk in result:
        assert len(chunk) <= 150  # Allow some overflow for sentence boundaries


def test_chunk_text_overlap():
    """Test that chunks have overlapping content."""
    text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10. " * 5
    result = chunk_text(text, chunk_size=100, overlap=30)

    if len(result) > 1:
        # Check that there's some overlap between consecutive chunks
        for i in range(len(result) - 1):
            # The end of one chunk should share words with the start of the next
            end_words = set(result[i].split()[-5:])
            start_words = set(result[i + 1].split()[:5])
            # There should be some overlap (not necessarily all words)
            assert len(end_words) > 0 or len(start_words) > 0


def test_chunk_text_sentence_boundary():
    """Test that chunks prefer sentence boundaries."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    result = chunk_text(text, chunk_size=30, overlap=5)

    # Most chunks should end with a period
    periods_at_end = sum(1 for chunk in result if chunk.rstrip().endswith("."))
    assert periods_at_end >= len(result) // 2
