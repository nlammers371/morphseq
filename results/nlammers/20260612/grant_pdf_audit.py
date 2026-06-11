#!/usr/bin/env python3
"""
Batch-audit PDFs for grant-submission problems:
  1) clickable external links / hidden hypertext annotations
  2) visible content intruding into required margins

Default settings are NIH-like: 0.5 inch margins on US letter-or-smaller pages.

Dependencies:
    python -m pip install pymupdf numpy

Example:
    python grant_pdf_audit.py /path/to/pdfs --recursive --report pdf_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple
from urllib.parse import urlparse

import fitz  # PyMuPDF
import numpy as np

URLISH_RE = re.compile(
    r"(https?://|ftp://|www\.|mailto:|doi\.org/|\bdoi\s*:|\b10\.\d{4,9}/|"
    r"\b[a-z0-9.-]+\.(gov|edu|org|com|net|io|ai)(/|\b))",
    re.IGNORECASE,
)
VISIBLE_URL_RE = re.compile(
    r"https?://\S+|ftp://\S+|www\.\S+|mailto:\S+|doi\.org/\S+|\bdoi\s*:\s*10\.\S+|\b10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    re.IGNORECASE,
)
RAW_LINK_PATTERNS = [
    ("/URI", re.compile(r"/URI\b")),
    ("/Launch", re.compile(r"/Launch\b")),
    ("/GoToR", re.compile(r"/GoToR\b")),
    ("/SubmitForm", re.compile(r"/SubmitForm\b")),
    ("/JavaScript", re.compile(r"/JavaScript\b")),
    ("/OpenAction", re.compile(r"/OpenAction\b")),
    ("/AA", re.compile(r"/AA\b")),
    ("/RichMedia", re.compile(r"/RichMedia\b")),
    ("/EmbeddedFile", re.compile(r"/EmbeddedFile\b")),
]
LINK_KIND_NAMES = {
    getattr(fitz, "LINK_NONE", 0): "NONE",
    getattr(fitz, "LINK_GOTO", 1): "GOTO_INTERNAL",
    getattr(fitz, "LINK_URI", 2): "URI",
    getattr(fitz, "LINK_LAUNCH", 3): "LAUNCH_FILE_OR_APP",
    getattr(fitz, "LINK_NAMED", 4): "NAMED_ACTION",
    getattr(fitz, "LINK_GOTOR", 5): "GOTO_REMOTE",
}


def rect_intersection_area(a: fitz.Rect, b: fitz.Rect) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def rect_to_str(r: fitz.Rect | None) -> str:
    if r is None:
        return ""
    return f"{r.x0:.2f},{r.y0:.2f},{r.x1:.2f},{r.y1:.2f}"


def rect_to_inches(r: fitz.Rect | None) -> str:
    if r is None:
        return ""
    return f"{r.x0/72:.3f},{r.y0/72:.3f},{r.x1/72:.3f},{r.y1/72:.3f}"


def clean_text(s: str, max_len: int = 240) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def text_under_rect(page: fitz.Page, rect: fitz.Rect) -> str:
    """Return words that overlap a rectangle. Useful for detecting hidden hypertext."""
    words = page.get_text("words") or []
    hits: List[Tuple[float, float, str]] = []
    for w in words:
        wr = fitz.Rect(w[0], w[1], w[2], w[3])
        area = rect_intersection_area(rect, wr)
        if area <= 0:
            continue
        word_area = max(1e-6, (wr.x1 - wr.x0) * (wr.y1 - wr.y0))
        # Include words with non-trivial overlap, but be permissive for skinny link boxes.
        if area / word_area >= 0.15 or area > 2:
            hits.append((wr.y0, wr.x0, str(w[4])))
    hits.sort()
    return clean_text(" ".join(t[2] for t in hits))


def target_for_link(link: Dict[str, Any]) -> str:
    pieces = []
    for k in ("uri", "file", "nameddest", "xref"):
        val = link.get(k)
        if val:
            pieces.append(f"{k}={val}")
    if "page" in link and link.get("kind") not in (getattr(fitz, "LINK_URI", 2), None):
        pieces.append(f"page={link.get('page')}")
    return "; ".join(pieces)


def visible_text_looks_like_actual_url(text: str, target: str) -> bool:
    """NIH-style heuristic: clickable target should be visible as URL/DOI/email text, not hidden behind words."""
    if not text:
        return False
    if URLISH_RE.search(text):
        return True
    # Sometimes the visible text is a bare hostname or DOI and the target contains the full URI.
    parsed = urlparse(target.replace("uri=", "", 1))
    host = (parsed.netloc or "").lower().removeprefix("www.")
    if host and host in text.lower():
        return True
    return False


def margin_sides_for_rect(rect: fitz.Rect, page_rect: fitz.Rect, margin_pt: float, tolerance_pt: float) -> List[str]:
    sides = []
    if rect.x0 < page_rect.x0 + margin_pt - tolerance_pt:
        sides.append("left")
    if rect.x1 > page_rect.x1 - margin_pt + tolerance_pt:
        sides.append("right")
    if rect.y0 < page_rect.y0 + margin_pt - tolerance_pt:
        sides.append("top")
    if rect.y1 > page_rect.y1 - margin_pt + tolerance_pt:
        sides.append("bottom")
    return sides


def add_issue(
    rows: List[Dict[str, Any]],
    pdf: Path,
    page_num: int | str,
    issue_type: str,
    severity: str,
    details: str,
    bbox: fitz.Rect | None = None,
    link_target: str = "",
    visible_text: str = "",
):
    rows.append(
        {
            "pdf": str(pdf),
            "page": page_num,
            "issue_type": issue_type,
            "severity": severity,
            "details": details,
            "bbox_points_x0_y0_x1_y1": rect_to_str(bbox),
            "bbox_inches_x0_y0_x1_y1": rect_to_inches(bbox),
            "link_or_target": clean_text(link_target, 500),
            "visible_text_in_link_rect": clean_text(visible_text, 300),
        }
    )


def scan_raw_pdf_objects(doc: fitz.Document, pdf: Path, rows: List[Dict[str, Any]]) -> None:
    """Find link/action-related structures that page.get_links may not expose."""
    try:
        xref_len = doc.xref_length()
    except Exception:
        return
    for xref in range(1, xref_len):
        try:
            obj = doc.xref_object(xref, compressed=False)
        except Exception:
            continue
        found = [label for label, rx in RAW_LINK_PATTERNS if rx.search(obj)]
        if not found:
            continue
        snippet = re.sub(r"\s+", " ", obj).strip()
        # Keep enough context to locate the object without dumping the PDF contents.
        snippet = snippet[:600] + ("…" if len(snippet) > 600 else "")
        add_issue(
            rows,
            pdf,
            page_num="object",
            issue_type="raw_pdf_action_or_external_object",
            severity="review",
            details=f"xref {xref} contains {', '.join(found)}. This may indicate links, actions, JavaScript, attachments, or embedded files.",
            link_target=snippet,
        )


def scan_links(page: fitz.Page, pdf: Path, page_index: int, rows: List[Dict[str, Any]]) -> None:
    for link in page.get_links() or []:
        kind = link.get("kind")
        kind_name = LINK_KIND_NAMES.get(kind, f"kind_{kind}")
        target = target_for_link(link)
        rect = fitz.Rect(link.get("from")) if link.get("from") else None
        visible = text_under_rect(page, rect) if rect is not None else ""

        # Internal jumps are usually harmless; everything else deserves review.
        is_internal = kind == getattr(fitz, "LINK_GOTO", 1)
        severity = "review" if is_internal else "flag"
        issue_type = "internal_link_annotation" if is_internal else "external_or_action_link_annotation"
        details = f"Clickable link annotation: {kind_name}."

        if not is_internal:
            if not visible_text_looks_like_actual_url(visible, target):
                issue_type = "hidden_or_hypertext_link_suspect"
                severity = "flag"
                details += " Target is not visibly printed as URL/DOI/email text in the link rectangle."
            else:
                details += " Target appears to be visible URL-like text, but confirm this attachment allows links/URLs."

        add_issue(rows, pdf, page_index + 1, issue_type, severity, details, rect, target, visible)


def scan_visible_url_text(page: fitz.Page, pdf: Path, page_index: int, rows: List[Dict[str, Any]]) -> None:
    text = page.get_text("text") or ""
    for match in VISIBLE_URL_RE.finditer(text):
        val = clean_text(match.group(0), 300)
        add_issue(
            rows,
            pdf,
            page_index + 1,
            "visible_url_or_doi_text",
            "review",
            "Visible URL/DOI/email-like text. NIH allows URLs only where specifically permitted; this is not necessarily clickable.",
            link_target=val,
        )


def scan_layout_margins(
    page: fitz.Page,
    pdf: Path,
    page_index: int,
    rows: List[Dict[str, Any]],
    margin_pt: float,
    tolerance_pt: float,
    check_drawings: bool,
) -> None:
    page_rect = page.rect

    # Text and image blocks from the PDF structure.
    for block in page.get_text("blocks") or []:
        if len(block) < 7:
            continue
        rect = fitz.Rect(block[0], block[1], block[2], block[3])
        btype = int(block[6])
        btext = block[4] if btype == 0 else ""
        if btype == 0 and not str(btext).strip():
            continue
        if rect.width <= 0 or rect.height <= 0:
            continue
        sides = margin_sides_for_rect(rect, page_rect, margin_pt, tolerance_pt)
        if not sides:
            continue
        label = "text" if btype == 0 else "image"
        add_issue(
            rows,
            pdf,
            page_index + 1,
            "margin_intrusion_layout",
            "flag",
            f"{label} block intersects required margin: {', '.join(sides)}.",
            rect,
            visible_text=clean_text(str(btext), 300),
        )

    # Optional: vector drawings. This can be noisy because full-page white backgrounds count as drawings.
    if check_drawings:
        for d in page.get_drawings() or []:
            rect = d.get("rect")
            if not rect:
                continue
            rect = fitz.Rect(rect)
            if rect.width < 1 or rect.height < 1:
                continue
            sides = margin_sides_for_rect(rect, page_rect, margin_pt, tolerance_pt)
            if not sides:
                continue
            add_issue(
                rows,
                pdf,
                page_index + 1,
                "margin_intrusion_vector_drawing",
                "review",
                f"Vector drawing intersects required margin: {', '.join(sides)}. Review visually; this can include harmless backgrounds.",
                rect,
            )


def scan_pixel_margins(
    page: fitz.Page,
    pdf: Path,
    page_index: int,
    rows: List[Dict[str, Any]],
    margin_pt: float,
    dpi: int,
    threshold: int,
    min_pixels: int,
) -> None:
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n > 3:
        arr = arr[:, :, :3]

    # A pixel is considered content if any channel is meaningfully non-white.
    # Increase threshold to be more sensitive; decrease to ignore faint scanner noise.
    content = np.any(arr < threshold, axis=2)

    # Define margin bands in rendered-pixel coordinates.
    left = int(round((page.rect.x0 + margin_pt) * scale))
    right = int(round((page.rect.x1 - margin_pt) * scale))
    top = int(round((page.rect.y0 + margin_pt) * scale))
    bottom = int(round((page.rect.y1 - margin_pt) * scale))

    outside = content.copy()
    outside[max(0, top): min(pix.height, bottom), max(0, left): min(pix.width, right)] = False
    ys, xs = np.nonzero(outside)
    if xs.size < min_pixels:
        return

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    rect = fitz.Rect(x0 / scale, y0 / scale, x1 / scale, y1 / scale)

    sides = []
    if np.any(outside[:, : max(0, left)]):
        sides.append("left")
    if np.any(outside[:, min(pix.width, right) :]):
        sides.append("right")
    if np.any(outside[: max(0, top), :]):
        sides.append("top")
    if np.any(outside[min(pix.height, bottom) :, :]):
        sides.append("bottom")

    add_issue(
        rows,
        pdf,
        page_index + 1,
        "margin_intrusion_rendered_pixels",
        "flag",
        f"Rendered non-white pixels found in required margin(s): {', '.join(sides)}; pixel_count={xs.size}. Review this page visually.",
        rect,
    )


def scan_page_size(doc: fitz.Document, pdf: Path, rows: List[Dict[str, Any]], tolerance_pt: float) -> None:
    for i, page in enumerate(doc):
        w, h = page.rect.width, page.rect.height
        small, large = sorted((w, h))
        # Standard letter is 612 x 792 pt. Landscape is okay if neither dimension exceeds letter in its orientation.
        if small > 612 + tolerance_pt or large > 792 + tolerance_pt:
            add_issue(
                rows,
                pdf,
                i + 1,
                "page_size_larger_than_letter",
                "flag",
                f"Page is {w/72:.2f} x {h/72:.2f} inches; NIH attachments should be no larger than 8.5 x 11 inches.",
            )


def audit_pdf(pdf: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(pdf)
    except Exception as e:
        add_issue(rows, pdf, "file", "cannot_open_pdf", "flag", f"Could not open PDF: {e}")
        return rows

    if doc.is_encrypted:
        add_issue(rows, pdf, "file", "encrypted_or_secured_pdf", "flag", "PDF is encrypted/secured; grant systems may not be able to process it.")

    margin_pt = float(args.margin_inches) * 72.0
    try:
        scan_page_size(doc, pdf, rows, args.tolerance_points)
        if args.raw_scan:
            scan_raw_pdf_objects(doc, pdf, rows)
        for page_index, page in enumerate(doc):
            scan_links(page, pdf, page_index, rows)
            if args.include_visible_url_text:
                scan_visible_url_text(page, pdf, page_index, rows)
            scan_layout_margins(page, pdf, page_index, rows, margin_pt, args.tolerance_points, args.check_drawings)
            if args.render_margin_check:
                scan_pixel_margins(page, pdf, page_index, rows, margin_pt, args.dpi, args.pixel_threshold, args.min_margin_pixels)
    except Exception as e:
        add_issue(rows, pdf, "file", "audit_error", "flag", f"Audit failed partway through: {e}")
    finally:
        doc.close()
    return rows


def find_pdfs(folder: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    yield from sorted(folder.glob(pattern))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch audit PDFs for hidden links/external actions and margin intrusions.")
    parser.add_argument("folder", type=Path, help="Folder containing PDFs")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders recursively")
    parser.add_argument("--report", type=Path, default=Path("pdf_audit_report.csv"), help="Output CSV path")
    parser.add_argument("--margin-inches", type=float, default=0.5, help="Required margin size in inches; default 0.5")
    parser.add_argument("--tolerance-points", type=float, default=1.0, help="Tolerance for layout bbox checks; default 1 pt")
    parser.add_argument("--raw-scan", action="store_true", default=True, help="Scan raw PDF objects for /URI, /Launch, /JavaScript, etc. Default on")
    parser.add_argument("--no-raw-scan", dest="raw_scan", action="store_false", help="Disable raw PDF object scan")
    parser.add_argument("--include-visible-url-text", action="store_true", help="Also report visible URL/DOI/email-like text, even if not clickable")
    parser.add_argument("--render-margin-check", action="store_true", default=True, help="Render pages and check non-white pixels in margins. Default on")
    parser.add_argument("--no-render-margin-check", dest="render_margin_check", action="store_false", help="Disable rendered-pixel margin check")
    parser.add_argument("--check-drawings", action="store_true", help="Also check vector drawing bounding boxes; can produce false positives")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for rendered-pixel margin check; default 150")
    parser.add_argument("--pixel-threshold", type=int, default=245, help="Pixels with any RGB channel below this count as content; default 245")
    parser.add_argument("--min-margin-pixels", type=int, default=50, help="Minimum non-white pixels in margins before flagging; default 50")
    args = parser.parse_args(argv)

    folder = args.folder.expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"ERROR: folder not found: {folder}", file=sys.stderr)
        return 2

    pdfs = list(find_pdfs(folder, args.recursive))
    if not pdfs:
        print(f"No PDFs found in {folder}")
        return 0

    all_rows: List[Dict[str, Any]] = []
    for pdf in pdfs:
        rel_pdf = pdf
        try:
            rel_pdf = pdf.relative_to(Path.cwd())
        except Exception:
            pass
        print(f"Auditing {rel_pdf}")
        all_rows.extend(audit_pdf(pdf, args))

    fieldnames = [
        "pdf",
        "page",
        "issue_type",
        "severity",
        "details",
        "bbox_points_x0_y0_x1_y1",
        "bbox_inches_x0_y0_x1_y1",
        "link_or_target",
        "visible_text_in_link_rect",
    ]
    report = args.report.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    flag_count = sum(1 for r in all_rows if r.get("severity") == "flag")
    review_count = sum(1 for r in all_rows if r.get("severity") == "review")
    print(f"\nWrote {len(all_rows)} findings to {report}")
    print(f"Summary: {flag_count} flags, {review_count} review items across {len(pdfs)} PDFs")
    if flag_count:
        print("Open the CSV and inspect rows with severity=flag first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
