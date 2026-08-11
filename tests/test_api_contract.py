"""API surface contract — offline.

Uses FastAPI's TestClient with the graph and Pinecone dependencies never touched,
so these run without keys or network.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_serves_the_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Legixo Q&amp;A" in r.text or "Legixo Q&A" in r.text


def test_ui_is_self_contained():
    """No CDN, no external fonts, no remote scripts — the page must render with
    only the app serving it."""
    body = client.get("/").text
    for pattern in ("src=\"http", "href=\"http", "//cdn.", "googleapis"):
        assert pattern not in body, f"UI references an external resource: {pattern}"


def test_ui_calls_the_documented_endpoint():
    """The page is a client over POST /ask, not a second implementation."""
    assert '"/ask"' in client.get("/").text


def test_ask_rejects_an_empty_question():
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_ask_requires_a_question_field():
    assert client.post("/ask", json={}).status_code == 422


def test_admin_ingest_rejects_a_bad_token():
    r = client.post("/admin/ingest", headers={"X-Ingest-Token": "wrong"})
    assert r.status_code == 401


def test_admin_ingest_rejects_a_missing_token():
    assert client.post("/admin/ingest").status_code == 401


@pytest.mark.parametrize("path", ["/docs", "/openapi.json"])
def test_interactive_docs_are_available(path):
    assert client.get(path).status_code == 200
