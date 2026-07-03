import io

import httpx
import pytest

from arxiv_reproducer import paper as paper_mod
from arxiv_reproducer.paper import fetch_paper, parse_arxiv_id


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
    state = {"feed": ATOM_FEED, "pdf_requests": 0}
    pdf = minimal_pdf_bytes()

    def handler(request):
        if request.url.path.startswith("/api/query"):
            return httpx.Response(200, text=state["feed"])
        state["pdf_requests"] += 1
        return httpx.Response(200, content=pdf)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        paper_mod.httpx, "Client", lambda **kw: real_client(transport=transport, **kw)
    )
    return state


class TestFetchPaper:
    def test_fetches_metadata_and_pdf(self, arxiv_transport, tmp_path):
        paper = fetch_paper("2301.12345", tmp_path)
        assert paper.arxiv_id == "2301.12345"
        assert paper.title == "Chaotic Dynamics of a Test System"
        assert paper.abstract == "We study a thing."
        assert paper.authors == ["A. Author", "B. Author"]
        assert paper.pdf_path.exists()
        assert isinstance(paper.full_text, str)  # blank page extracts to empty text

    def test_cached_pdf_is_not_redownloaded(self, arxiv_transport, tmp_path):
        fetch_paper("2301.12345", tmp_path)
        fetch_paper("2301.12345", tmp_path)
        assert arxiv_transport["pdf_requests"] == 1

    def test_unknown_id_raises_clear_error(self, arxiv_transport, tmp_path):
        arxiv_transport["feed"] = EMPTY_FEED
        with pytest.raises(ValueError, match="no entry"):
            fetch_paper("9999.99999", tmp_path)
