"""Fetch arXiv papers: metadata via the arXiv API, full text via PDF extraction."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx
from pypdf import PdfReader

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

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
        meta = client.get(ARXIV_API, params={"id_list": arxiv_id, "max_results": 1})
        meta.raise_for_status()
        entry = ET.fromstring(meta.text).find("atom:entry", ATOM_NS)
        if entry is None:
            raise ValueError(f"arXiv API returned no entry for {arxiv_id}")

        title = entry.findtext("atom:title", "", ATOM_NS).strip()
        abstract = entry.findtext("atom:summary", "", ATOM_NS).strip()
        authors = [
            a.findtext("atom:name", "", ATOM_NS).strip()
            for a in entry.findall("atom:author", ATOM_NS)
        ]

        pdf_path = workdir / f"{arxiv_id.replace('/', '_')}.pdf"
        if not pdf_path.exists():
            pdf = client.get(ARXIV_PDF.format(arxiv_id=arxiv_id))
            pdf.raise_for_status()
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
    reader = PdfReader(pdf_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
