"""Structure-aware Markdown chunker with token-based sizing."""

from __future__ import annotations

import re

import tiktoken

from kisan.core.logging import logger

# Regex patterns for Markdown structure
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|.+\|$", re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^[\s]*[-*+]\s|^[\s]*\d+[.)]\s", re.MULTILINE)


class MarkdownChunker:
    """Recursively chunks Markdown text respecting structure boundaries.

    Splitting hierarchy:
      1. ## (H2) — major sections
      2. ### (H3) — subsections
      3. Paragraph boundaries (double newline)
      4. Sentence boundaries
    Tables and bullet-point groups are kept atomic.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        model_name: str = "text-embedding-3-small",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._enc = tiktoken.encoding_for_model(model_name)

    def token_count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def chunk(self, markdown: str, source: str = "") -> list[dict]:
        """Chunk a full Markdown document into enriched metadata dicts.

        Returns list of dicts with keys:
          content, source, scheme_name, section_header, section_hierarchy,
          content_type, page_numbers, chunk_index
        """
        scheme_name = self._extract_scheme_name(markdown)
        sections = self._split_into_sections(markdown)
        chunks: list[dict] = []
        chunk_index = 0

        for section_header, section_hierarchy, section_text in sections:
            section_chunks = self._chunk_section(section_text)
            for text in section_chunks:
                content_type = self._detect_content_type(text)
                page_numbers = self._extract_page_numbers(text)
                # Strip page comment markers from final content
                clean_text = re.sub(r"<!--\s*page\s+\d+\s*-->\n?", "", text).strip()
                if not clean_text:
                    continue
                chunks.append({
                    "content": clean_text,
                    "source": source,
                    "scheme_name": scheme_name,
                    "section_header": section_header,
                    "section_hierarchy": section_hierarchy,
                    "content_type": content_type,
                    "page_numbers": page_numbers,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

        logger.info(
            f"Chunked document into {len(chunks)} chunks "
            f"(target {self.chunk_size} tokens, {self.chunk_overlap} overlap)"
        )
        return chunks

    # --- Internal helpers ---

    def _extract_scheme_name(self, markdown: str) -> str:
        """Extract scheme name from the first H1 heading, or return empty."""
        match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
        # Try the first non-empty line as a fallback
        for line in markdown.splitlines():
            line = line.strip()
            if line and not line.startswith("<!--"):
                return line[:120]
        return ""

    def _split_into_sections(
        self, markdown: str
    ) -> list[tuple[str, str, str]]:
        """Split markdown into (section_header, section_hierarchy, text) tuples.

        Uses H2 headings as primary splits. If no H2s exist, treats the
        entire document as one section.
        """
        # Find all headings with their levels and positions
        headings = list(_HEADING_RE.finditer(markdown))

        if not headings:
            return [("", "", markdown)]

        # Build hierarchy: track the most recent heading at each level
        current_hierarchy: dict[int, str] = {}
        sections: list[tuple[str, str, str]] = []

        # Content before the first heading
        first_pos = headings[0].start()
        if first_pos > 0:
            preamble = markdown[:first_pos].strip()
            if preamble:
                sections.append(("", "", preamble))

        for i, match in enumerate(headings):
            level = len(match.group(1))  # number of # chars
            heading_text = match.group(2).strip()
            current_hierarchy[level] = heading_text
            # Clear deeper levels
            for lvl in list(current_hierarchy):
                if lvl > level:
                    del current_hierarchy[lvl]

            hierarchy = " > ".join(
                current_hierarchy[lvl]
                for lvl in sorted(current_hierarchy)
            )

            # Section text runs from this heading to the next
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
            section_text = markdown[start:end].strip()

            if section_text:
                sections.append((heading_text, hierarchy, section_text))

        return sections

    def _chunk_section(self, text: str) -> list[str]:
        """Recursively chunk a section to fit within token limits."""
        if self.token_count(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Try splitting by H3 headers
        parts = self._split_by_pattern(text, r"\n(?=###\s)")
        if len(parts) > 1:
            return self._merge_or_recurse(parts)

        # Try splitting by paragraphs (double newline)
        parts = self._split_by_paragraphs(text)
        if len(parts) > 1:
            return self._merge_or_recurse(parts)

        # Last resort: split by sentences
        return self._split_by_sentences(text)

    def _split_by_pattern(self, text: str, pattern: str) -> list[str]:
        """Split text by regex pattern, keeping the delimiter with the following part."""
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split text on double newlines, keeping tables and lists atomic."""
        raw_blocks = re.split(r"\n\n+", text)
        blocks: list[str] = []
        current_atomic: list[str] = []

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            is_table = bool(_TABLE_ROW_RE.search(block))
            is_list = bool(_LIST_ITEM_RE.match(block))

            if is_table or is_list:
                # Accumulate consecutive table/list blocks as atomic
                current_atomic.append(block)
            else:
                if current_atomic:
                    blocks.append("\n\n".join(current_atomic))
                    current_atomic = []
                blocks.append(block)

        if current_atomic:
            blocks.append("\n\n".join(current_atomic))

        return blocks

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text by sentence boundaries as a last resort."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = self.token_count(sentence)
            if current_tokens + s_tokens > self.chunk_size and current:
                chunks.append(" ".join(current))
                # Overlap: keep tail sentences
                overlap_tokens = 0
                overlap_start = len(current)
                for j in range(len(current) - 1, -1, -1):
                    overlap_tokens += self.token_count(current[j])
                    if overlap_tokens >= self.chunk_overlap:
                        overlap_start = j
                        break
                current = current[overlap_start:]
                current_tokens = sum(self.token_count(s) for s in current)

            current.append(sentence)
            current_tokens += s_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _merge_or_recurse(self, parts: list[str]) -> list[str]:
        """Merge small parts together; recursively chunk large parts."""
        chunks: list[str] = []
        buffer = ""
        buffer_tokens = 0

        for part in parts:
            part_tokens = self.token_count(part)

            if part_tokens > self.chunk_size:
                # Flush buffer first
                if buffer.strip():
                    chunks.append(buffer.strip())
                    buffer = ""
                    buffer_tokens = 0
                # Recurse on oversized part
                chunks.extend(self._chunk_section(part))
                continue

            if buffer_tokens + part_tokens > self.chunk_size:
                # Flush buffer
                if buffer.strip():
                    chunks.append(buffer.strip())
                # Start new buffer with overlap from end of previous
                if self.chunk_overlap > 0 and buffer:
                    overlap_text = self._get_tail_tokens(buffer, self.chunk_overlap)
                    buffer = overlap_text + "\n\n" + part
                    buffer_tokens = self.token_count(buffer)
                else:
                    buffer = part
                    buffer_tokens = part_tokens
            else:
                buffer = (buffer + "\n\n" + part).strip() if buffer else part
                buffer_tokens += part_tokens

        if buffer.strip():
            chunks.append(buffer.strip())

        return chunks

    def _get_tail_tokens(self, text: str, n_tokens: int) -> str:
        """Get the last ~n_tokens worth of text."""
        tokens = self._enc.encode(text)
        if len(tokens) <= n_tokens:
            return text
        return self._enc.decode(tokens[-n_tokens:])

    def _detect_content_type(self, text: str) -> str:
        """Classify chunk content type."""
        if _TABLE_ROW_RE.search(text):
            return "table"
        if _LIST_ITEM_RE.search(text):
            return "list"
        if re.search(r'"[^"]+"\s+means', text, re.IGNORECASE):
            return "definition"
        return "paragraph"

    def _extract_page_numbers(self, text: str) -> list[int]:
        """Extract page numbers from PyMuPDF fallback page markers."""
        pages = re.findall(r"<!--\s*page\s+(\d+)\s*-->", text)
        return sorted(set(int(p) for p in pages)) if pages else []
