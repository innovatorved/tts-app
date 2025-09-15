import re
import logging

logger = logging.getLogger(__name__)


def smart_split_text(text: str, split_pattern: str = r"\n\n+|\r\n\r\n+|\n\s*\n+|[.!?]\s") -> list[str]:
    """
    Intelligently splits text into segments for TTS.
    If text is short (less than 5 lines or 500 characters), returns as single segment.
    Otherwise, splits using the provided pattern.
    """
    if not text or not text.strip():
        return []
    
    # Count lines and characters
    lines = text.split('\n')
    line_count = len([line for line in lines if line.strip()])
    char_count = len(text)
    
    # If text is short, don't split
    if line_count <= 5 or char_count <= 500:
        logger.info(f"Text is short ({line_count} lines, {char_count} chars), not splitting.")
        return [text.strip()]
    
    # Otherwise, split intelligently
    try:
        parts = re.split(split_pattern, text)
        # Clean and filter
        cleaned = [p.strip() for p in parts if p and p.strip()]
        if not cleaned:
            return [text.strip()]
        
        # Merge very short segments with previous
        merged = []
        current = ""
        for part in cleaned:
            if len(current) + len(part) < 200:  # If combined is still short, merge
                current += " " + part if current else part
            else:
                if current:
                    merged.append(current)
                current = part
        if current:
            merged.append(current)
        
        logger.info(f"Smart split into {len(merged)} segments.")
        return merged
    except re.error:
        logger.warning("Invalid split pattern, using fallback.")
        return [text.strip()]


def split_text_into_chunks(
    full_text: str, max_paragraphs_per_chunk: int = 30
) -> list[str]:
    """Splits text into larger chunks based on paragraph count."""
    if not full_text or not full_text.strip():
        return []

    # Normalize newlines then split by patterns that likely indicate paragraph breaks
    normalized_text = full_text.replace("\r\n", "\n").replace("\r", "\n")
    # Split by two or more newlines, possibly with spaces in between
    paragraphs = re.split(r"\n\s*\n+", normalized_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if (
        not paragraphs
    ):  # If no clear paragraph breaks, treat as one large paragraph block
        if normalized_text.strip():
            paragraphs = [normalized_text.strip()]
        else:
            return []

    chunks = []
    current_chunk_paragraphs = []
    for i, para in enumerate(paragraphs):
        current_chunk_paragraphs.append(para)
        if len(current_chunk_paragraphs) >= max_paragraphs_per_chunk or (i + 1) == len(
            paragraphs
        ):
            chunks.append(
                "\n\n".join(current_chunk_paragraphs)
            )  # Rejoin with standard double newline
            current_chunk_paragraphs = []

    if not chunks and full_text.strip():  # Ensure at least one chunk if there's text
        chunks = [
            "\n\n".join(paragraphs)
        ]  # Rejoin all found paragraphs if they didn't form a chunk

    logger.info(
        f"Split text into {len(chunks)} chunks, targeting up to {max_paragraphs_per_chunk} paragraphs per chunk."
    )
    return chunks
