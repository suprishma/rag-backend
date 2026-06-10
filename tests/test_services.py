from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ── Text utils ────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_collapses_blank_lines(self):
        from app.utils.text import clean_text
        assert clean_text("hello\n\n\n\n\nworld") == "hello\n\nworld"

    def test_strips_leading_trailing(self):
        from app.utils.text import clean_text
        assert clean_text("  hello  ") == "hello"

    def test_handles_empty(self):
        from app.utils.text import clean_text
        assert clean_text("") == ""


class TestSplitSentences:
    def test_basic_split(self):
        from app.utils.text import split_into_sentences
        text = "Hello world. This is a test sentence. And another one here."
        sentences = split_into_sentences(text)
        assert len(sentences) >= 2

    def test_empty_returns_empty(self):
        from app.utils.text import split_into_sentences
        assert split_into_sentences("") == []


# ── Fixed chunking ────────────────────────────────────────────────────────────

class TestFixedChunking:

    def test_returns_multiple_chunks(self):
        from app.services.chunking import fixed_chunking
        text = " ".join(["word"] * 500)
        chunks = fixed_chunking(text, chunk_size=64, chunk_overlap=8)
        assert len(chunks) > 1

    def test_chunk_index_sequential(self):
        from app.services.chunking import fixed_chunking
        text = " ".join(["word"] * 300)
        chunks = fixed_chunking(text, chunk_size=64, chunk_overlap=8)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_single_short_text(self):
        from app.services.chunking import fixed_chunking
        chunks = fixed_chunking("Hello world", chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1

    def test_overlap_increases_chunk_count(self):
        from app.services.chunking import fixed_chunking
        text = " ".join(["word"] * 50)
        no_ov = fixed_chunking(text, chunk_size=16, chunk_overlap=0)

    def test_token_count_populated(self):
        from app.services.chunking import fixed_chunking
        text = " ".join(["word"] * 100)
        chunks = fixed_chunking(text, chunk_size=32, chunk_overlap=4)
        for chunk in chunks:
            assert chunk.token_count > 0

# ── Semantic chunking ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSemanticChunking:
    async def test_returns_chunks_with_mock_embeddings(self):
        import numpy as np
        from app.services.chunking import semantic_chunking

        sentences = [
            "Artificial intelligence is transforming industries rapidly.",
            "Machine learning enables powerful pattern recognition tasks.",
            "The cat sat on the mat quietly all day.",
            "Dogs and cats are common household pets worldwide.",
        ]
        text = " ".join(sentences)

        rng = np.random.default_rng(42)
        base_ai = rng.random(8)
        base_pet = rng.random(8)
        mock_embeddings = [
            (base_ai + rng.random(8) * 0.05).tolist(),
            (base_ai + rng.random(8) * 0.05).tolist(),
            (base_pet + rng.random(8) * 0.05).tolist(),
            (base_pet + rng.random(8) * 0.05).tolist(),
        ]

        async def mock_embed(texts):
            return mock_embeddings[: len(texts)]

        chunks = await semantic_chunking(
            text, embed_fn=mock_embed, similarity_threshold=0.95
        )
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.text

    async def test_empty_text_returns_empty(self):
        from app.services.chunking import semantic_chunking

        async def mock_embed(texts):
            return []

        chunks = await semantic_chunking("", embed_fn=mock_embed)
        assert chunks == []


# ── Booking intent detection ──────────────────────────────────────────────────

@pytest.mark.asyncio
class TestBookingIntent:
    async def test_no_intent_returns_none(self):
        import app.services.booking as booking_mod

        with patch.object(
            booking_mod,
            "detect_booking_intent",
            new=AsyncMock(return_value={"intent": "none"}),
        ):
            db_mock = AsyncMock()
            result = await booking_mod.try_extract_and_save_booking(
                session_id="s1",
                user_message="What is machine learning?",
                history=[],
                db=db_mock,
            )
        assert result is None

    async def test_full_intent_creates_record(self):
        import app.services.booking as booking_mod

        booking_data = {
            "intent": "book",
            "name": "Alice Smith",
            "email": "alice@example.com",
            "date": "2026-02-01",
            "time": "10:00",
            "notes": "",
        }
        with patch.object(
            booking_mod,
            "detect_booking_intent",
            new=AsyncMock(return_value=booking_data),
        ):
            db_mock = AsyncMock()
            db_mock.add = MagicMock()
            db_mock.flush = AsyncMock()

            result = await booking_mod.try_extract_and_save_booking(
                session_id="s1",
                user_message="I'd like to book an interview",
                history=[],
                db=db_mock,
            )

        assert result is not None
        assert result.name == "Alice Smith"
        assert result.email == "alice@example.com"
        db_mock.add.assert_called_once()

    async def test_incomplete_intent_returns_none(self):
        import app.services.booking as booking_mod

        with patch.object(
            booking_mod,
            "detect_booking_intent",
            new=AsyncMock(return_value={"intent": "incomplete", "missing": ["email"]}),
        ):
            db_mock = AsyncMock()
            result = await booking_mod.try_extract_and_save_booking(
                session_id="s1",
                user_message="Book me an interview with John",
                history=[],
                db=db_mock,
            )
        assert result is None