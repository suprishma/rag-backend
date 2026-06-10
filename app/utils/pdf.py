from __future__ import annotations

import io

import fitz  # PyMuPDF

from app.core.logging import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(data: bytes) -> str:
    """Extract plain text from a PDF byte stream using PyMuPDF."""
    text_parts: list[str] = []
    with fitz.open(stream=io.BytesIO(data), filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(page_text)
    full_text = "\n".join(text_parts)
    logger.info("pdf.extracted", pages=len(text_parts), chars=len(full_text))
    return full_text


def extract_text_from_txt(data: bytes) -> str:
    """Decode a plain-text byte stream, trying UTF-8 then Latin-1."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")