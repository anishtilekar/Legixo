from app.chunking import chunk_markdown, count_tokens
from app.config import CHUNK_HARD_CEILING_TOKENS

SAMPLE = """# Employment agreement excerpt — Bluecrest Analytics (fiction)

**Employee:** Priya Nambiar
**Employer:** Bluecrest Analytics LLP

## Notice period

Either party may end this agreement by giving **60 days** written notice.

## Non-compete

For **12 months** after leaving, the employee may not work for a competitor.
"""


def test_splits_on_headings():
    chunks = chunk_markdown(SAMPLE)
    heading_paths = {c.heading_path for c in chunks}
    assert any("Notice period" in h for h in heading_paths)
    assert any("Non-compete" in h for h in heading_paths)


def test_chunk_index_is_sequential():
    chunks = chunk_markdown(SAMPLE)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_never_exceeds_hard_ceiling():
    long_section = "# Title\n\n## Body\n\n" + ("word " * 2000)
    chunks = chunk_markdown(long_section, chunk_tokens=350, overlap_tokens=60)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= CHUNK_HARD_CEILING_TOKENS


def test_overlap_produces_shared_content():
    long_section = "# Title\n\n## Body\n\n" + " ".join(f"word{i}" for i in range(1500))
    chunks = chunk_markdown(long_section, chunk_tokens=350, overlap_tokens=60)
    assert len(chunks) >= 2
    # some tail tokens of chunk 0 should reappear at the head of chunk 1
    tail = set(chunks[0].text.split()[-10:])
    head = set(chunks[1].text.split()[:30])
    assert tail & head


def test_empty_and_whitespace_sections_are_skipped():
    md = "# Title\n\n## Empty\n\n \n\n## Real\n\nSome content here.\n"
    chunks = chunk_markdown(md)
    assert all(c.text.strip() for c in chunks)
    assert any("Real" in c.heading_path for c in chunks)


def test_chunk_tokens_above_ceiling_raises():
    import pytest

    with pytest.raises(AssertionError):
        chunk_markdown(SAMPLE, chunk_tokens=500, overlap_tokens=60)


def test_embedding_text_prefixes_heading_context():
    """A chunk of bare figures must still carry the entity it belongs to, or a
    query naming those parties cannot retrieve it."""
    from app.chunking import embedding_text

    body = "- Damages: 8,50,000\n- Interest: 9% per year"
    heading = "Judgment summary - CV-2025-1190 > Relief granted"
    out = embedding_text(heading, body)

    assert out.startswith(heading)
    assert body in out
    assert "CV-2025-1190" in out, "case number must reach the vector"


def test_embedding_text_without_heading_is_unchanged():
    from app.chunking import embedding_text

    assert embedding_text("", "raw body") == "raw body"
    assert embedding_text("   ", "raw body") == "raw body"
