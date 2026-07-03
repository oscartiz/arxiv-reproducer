import io

import httpx
import pytest

from arxiv_reproducer import paper as paper_mod
from arxiv_reproducer import retry as retry_mod
from arxiv_reproducer.paper import PdfExtractionError, _extract_text, fetch_paper, parse_arxiv_id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2301.12345", "2301.12345"),
        ("2301.12345v2", "2301.12345v2"),
        ("https://arxiv.org/abs/2301.12345", "2301.12345"),
        ("https://arxiv.org/pdf/2301.12345v1", "2301.12345v1"),
        ("hep-th/9901001", "hep-th/9901001"),
        ("  2301.12345  ", "2301.12345"),
    ],
)
def test_parse_arxiv_id(raw, expected):
    assert parse_arxiv_id(raw) == expected


def test_parse_arxiv_id_rejects_garbage():
    with pytest.raises(ValueError):
        parse_arxiv_id("not a paper")


ATOM_FEED = """\
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Chaotic Dynamics of a Test System</title>
    <summary>We study a thing.</summary>
    <author><name>A. Author</name></author>
    <author><name>B. Author</name></author>
  </entry>
</feed>"""

EMPTY_FEED = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'

# Captured before any monkeypatching so tests can build real clients.
REAL_HTTPX_CLIENT = httpx.Client


def minimal_pdf_bytes() -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def arxiv_transport(monkeypatch):
    """Route paper.py's HTTP calls through an in-memory fake arXiv."""
    state = {"feed": ATOM_FEED, "pdf_requests": 0, "meta_requests": 0, "fail_first_n": 0}
    pdf = minimal_pdf_bytes()

    def handler(request):
        if request.url.path.startswith("/api/query"):
            state["meta_requests"] += 1
            if state["meta_requests"] <= state["fail_first_n"]:
                return httpx.Response(503)
            return httpx.Response(200, text=state["feed"])
        state["pdf_requests"] += 1
        return httpx.Response(200, content=pdf)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        paper_mod.httpx, "Client", lambda **kw: REAL_HTTPX_CLIENT(transport=transport, **kw)
    )
    # Real text extraction is exercised in TestExtractText; the fixture PDF is
    # a blank page, which would (correctly) be refused as scanned/imagelike.
    monkeypatch.setattr(paper_mod, "_extract_text", lambda p: "extracted text " * 100)
    monkeypatch.setattr(retry_mod.time, "sleep", lambda s: None)
    return state


class TestFetchPaper:
    def test_fetches_metadata_and_pdf(self, arxiv_transport, tmp_path):
        paper = fetch_paper("2301.12345", tmp_path)
        assert paper.arxiv_id == "2301.12345"
        assert paper.title == "Chaotic Dynamics of a Test System"
        assert paper.abstract == "We study a thing."
        assert paper.authors == ["A. Author", "B. Author"]
        assert paper.pdf_path.exists()
        assert "extracted text" in paper.full_text

    def test_cached_pdf_is_not_redownloaded(self, arxiv_transport, tmp_path):
        fetch_paper("2301.12345", tmp_path)
        fetch_paper("2301.12345", tmp_path)
        assert arxiv_transport["pdf_requests"] == 1

    def test_unknown_id_raises_clear_error(self, arxiv_transport, tmp_path):
        arxiv_transport["feed"] = EMPTY_FEED
        with pytest.raises(ValueError, match="no entry"):
            fetch_paper("9999.99999", tmp_path)

    def test_transient_api_errors_are_retried(self, arxiv_transport, tmp_path):
        arxiv_transport["fail_first_n"] = 2
        paper = fetch_paper("2301.12345", tmp_path)
        assert paper.title == "Chaotic Dynamics of a Test System"
        assert arxiv_transport["meta_requests"] == 3  # two 503s, then success

    def test_persistent_outage_finally_raises(self, arxiv_transport, tmp_path):
        arxiv_transport["fail_first_n"] = 99
        with pytest.raises(httpx.HTTPStatusError):
            fetch_paper("2301.12345", tmp_path)

    def test_client_error_is_not_retried(self, arxiv_transport, monkeypatch, tmp_path):
        def handler(request):
            arxiv_transport["meta_requests"] += 1
            return httpx.Response(404)

        monkeypatch.setattr(
            paper_mod.httpx,
            "Client",
            lambda **kw: REAL_HTTPX_CLIENT(transport=httpx.MockTransport(handler), **kw),
        )
        with pytest.raises(httpx.HTTPStatusError):
            fetch_paper("2301.12345", tmp_path)
        assert arxiv_transport["meta_requests"] == 1


class TestExtractText:
    def test_blank_pdf_is_refused_as_scannedlike(self, tmp_path):
        pdf = tmp_path / "blank.pdf"
        pdf.write_bytes(minimal_pdf_bytes())
        with pytest.raises(PdfExtractionError, match="scanned"):
            _extract_text(pdf)

    def test_corrupt_pdf_raises_extraction_error_not_pypdf_internals(self, tmp_path):
        pdf = tmp_path / "corrupt.pdf"
        pdf.write_bytes(b"%PDF-1.4 this is not really a pdf at all")
        with pytest.raises(PdfExtractionError):
            _extract_text(pdf)
