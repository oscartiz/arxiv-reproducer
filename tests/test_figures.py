from PIL import Image

from arxiv_reproducer.figures import extract_paper_figures


def make_pdf(path, sizes, color="red"):
    """A real PDF with one embedded raster image per page."""
    pages = [Image.new("RGB", size, color) for size in sizes]
    pages[0].save(path, format="PDF", save_all=True, append_images=pages[1:])


class TestExtraction:
    def test_extracts_figure_sized_images_as_pngs(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        make_pdf(pdf, [(200, 150)])
        written = extract_paper_figures(pdf, tmp_path / "figs")
        assert [path.name for path in written] == ["page01-img01.png"]
        with Image.open(written[0]) as image:
            assert image.size == (200, 150)

    def test_one_figure_per_page_numbering(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        make_pdf(pdf, [(200, 150), (300, 200)])
        written = extract_paper_figures(pdf, tmp_path / "figs")
        assert [path.name for path in written] == ["page01-img01.png", "page02-img01.png"]

    def test_tiny_images_are_skipped_and_dir_not_created(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        make_pdf(pdf, [(8, 8)])  # 64 px — a logo, not a figure
        assert extract_paper_figures(pdf, tmp_path / "figs") == []
        assert not (tmp_path / "figs").exists()

    def test_figure_cap_stops_extraction(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        make_pdf(pdf, [(200, 150)] * 5)
        written = extract_paper_figures(pdf, tmp_path / "figs", max_figures=2)
        assert len(written) == 2


class TestBestEffortTolerance:
    """Figure extraction must never break a reproduction run."""

    def test_garbage_pdf_returns_empty(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-fake but not really a pdf")
        assert extract_paper_figures(pdf, tmp_path / "figs") == []

    def test_missing_pdf_returns_empty(self, tmp_path):
        assert extract_paper_figures(tmp_path / "nope.pdf", tmp_path / "figs") == []
