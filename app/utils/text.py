from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Collapse excessive whitespace and remove non-printable characters."""
    # Replace various unicode line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove non-printable except newline/tab
    text = re.sub(r"[^\S\n\t ]+", " ", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter using punctuation boundaries."""
    # Split on . ! ? followed by whitespace or end
    raw = re.split(r"(?<=[.!?])\s+", text)
    # Rejoin very short fragments (abbreviations etc.)
    sentences: list[str] = []
    buffer = ""
    for part in raw:
        buffer = (buffer + " " + part).strip() if buffer else part
        if len(buffer.split()) >= 5:
            sentences.append(buffer)
            buffer = ""
    if buffer:
        sentences.append(buffer)
    return [s for s in sentences if s.strip()]