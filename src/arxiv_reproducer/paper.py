"""Fetch arXiv papers: metadata via the arXiv API, full text via PDF extraction."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pypdf import PdfReader

from .logs import get_logger
from .retry import retry_with_backoff

logger = get_logger("paper")

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Below this, the "text" is almost certainly a scanned/image PDF, not prose.
MIN_EXTRACTED_CHARS = 500


class PdfExtractionError(RuntimeError):
    """The PDF could not be turned into usable text."""

# Matches both new-style (2301.12345, optionally with version) and
# old-style (hep-th/9901001) identifiers, bare or inside an arxiv.org URL.
_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?([a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(v\d+)?")


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    full_text: str
    pdf_path: Path


def parse_arxiv_id(raw: str) -> str:
    """Normalize a user-supplied paper reference (ID or URL) to a bare arXiv ID."""
    match = _ID_RE.search(raw.strip())
    if not match:
        raise ValueError(f"Could not parse an arXiv ID from: {raw!r}")
    return match.group(1) + (match.group(2) or "")


def fetch_paper(raw_id: str, workdir: Path) -> Paper:
    """Download metadata and PDF for a paper, extract its text, and return a Paper."""
    arxiv_id = parse_arxiv_id(raw_id)
    workdir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=60, follow_redirects=True) as client:

        def get(url: str, **kwargs: Any) -> httpx.Response:
            def once() -> httpx.Response:
                response = client.get(url, **kwargs)
                response.raise_for_status()
                return response

            def log_retry(attempt: int, delay: float, exc: Exception) -> None:
                logger.warning(
                    "transient failure fetching %s (attempt %d, retrying in %.1fs): %s",
                    url, attempt, delay, exc,
                )

            return retry_with_backoff(once, on_retry=log_retry)

        meta = get(ARXIV_API, params={"id_list": arxiv_id, "max_results": 1})
        entry = ET.fromstring(meta.text).find("atom:entry", ATOM_NS)
        if entry is None:
            raise ValueError(f"arXiv API returned no entry for {arxiv_id}")

        title = entry.findtext("atom:title", "", ATOM_NS).strip()
        abstract = entry.findtext("atom:summary", "", ATOM_NS).strip()
        authors = [
            a.findtext("atom:name", "", ATOM_NS).strip()
            for a in entry.findall("atom:author", ATOM_NS)
        ]

        pdf_path = workdir / "paper.pdf"
        if not pdf_path.exists():
            pdf = get(ARXIV_PDF.format(arxiv_id=arxiv_id))
            pdf_path.write_bytes(pdf.content)

    full_text = _extract_text(pdf_path)
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=authors,
        full_text=full_text,
        pdf_path=pdf_path,
    )


def _extract_text(pdf_path: Path) -> str:
    """Extract text, tolerating per-page failures; refuse unusable PDFs."""
    try:
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # pypdf page-level parsing is fragile
                pages.append("")
        text = "\n\n".join(pages)
    except PdfExtractionError:
        raise
    except Exception as exc:
        raise PdfExtractionError(f"Could not parse PDF {pdf_path.name}: {exc}") from exc
    if len(text.strip()) < MIN_EXTRACTED_CHARS:
        raise PdfExtractionError(
            f"Extracted only {len(text.strip())} characters from {pdf_path.name} — "
            "this looks like a scanned/image PDF, which this tool cannot process."
        )
    return text
