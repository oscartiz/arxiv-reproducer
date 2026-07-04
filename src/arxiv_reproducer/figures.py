"""Extract the paper's own figures from its PDF, for side-by-side comparison.

Embedded raster images are pulled out of the PDF into the run workspace
(paper-figures/pageNN-imgNN.png) so the report can show the paper's original
figure next to the agent's regenerated one. Extraction is strictly
best-effort: a PDF that yields no images, or a page/image that fails to
decode, must never break a reproduction run.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .logs import get_logger

logger = get_logger("figures")

# Below this many pixels an image is almost certainly a logo, an equation
# rendered as a bitmap, or page decoration — not a figure worth comparing.
MIN_FIGURE_PIXELS = 10_000
MAX_FIGURES = 24


def extract_paper_figures(
    pdf_path: Path,
    out_dir: Path,
    max_figures: int = MAX_FIGURES,
    min_pixels: int = MIN_FIGURE_PIXELS,
) -> list[Path]:
    """Save the PDF's embedded raster images as PNGs; return the written paths.

    Never raises: any failure — unreadable PDF, undecodable image — is logged
    and skipped. The out_dir is only created if at least one figure survives.
    """
    written: list[Path] = []
    try:
        reader = PdfReader(pdf_path)
        pages = list(reader.pages)
    except Exception as exc:  # any pypdf failure means "no figures"
        logger.warning("could not open %s for figure extraction: %s", pdf_path, exc)
        return written

    for page_number, page in enumerate(pages, start=1):
        try:
            images = list(page.images)
        except Exception as exc:
            logger.debug("page %d: image enumeration failed: %s", page_number, exc)
            continue
        for image_number, image_file in enumerate(images, start=1):
            if len(written) >= max_figures:
                logger.info("figure cap (%d) reached, stopping extraction", max_figures)
                return written
            try:
                image = image_file.image
                if image is None or image.width * image.height < min_pixels:
                    continue
                if image.mode not in ("RGB", "RGBA", "L", "LA", "P", "1"):
                    image = image.convert("RGB")  # e.g. CMYK, unsupported by PNG
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir / f"page{page_number:02d}-img{image_number:02d}.png"
                image.save(target, format="PNG")
                written.append(target)
            except Exception as exc:
                logger.debug(
                    "page %d image %d: extraction failed: %s", page_number, image_number, exc
                )
    return written
