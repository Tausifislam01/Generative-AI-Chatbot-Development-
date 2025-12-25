from __future__ import annotations
import os
from pathlib import Path
from typing import List
from io import BytesIO
import pytesseract
from PIL import Image
from pdf2image import convert_from_path


def _configure_binaries() -> None:
    tcmd = os.getenv("TESSERACT_CMD")
    if tcmd:
        pytesseract.pytesseract.tesseract_cmd = tcmd


def ocr_image_file(file_path: Path, lang: str = "eng") -> str:
    _configure_binaries()
    img = Image.open(str(file_path))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return (pytesseract.image_to_string(img, lang=lang) or "").strip()


def ocr_pdf_file(
    file_path: Path,
    lang: str = "eng",
    max_pages: int = 5,
    dpi: int = 250,
) -> str:
    _configure_binaries()


    poppler_path = os.getenv("POPPLER_PATH")

    kwargs = {}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    pages = convert_from_path(str(file_path), dpi=dpi, first_page=1, last_page=max_pages, **kwargs)

    texts: List[str] = []
    for i, page_img in enumerate(pages, start=1):
        page_text = pytesseract.image_to_string(page_img, lang=lang) or ""
        texts.append(f"[PAGE {i}]\n{page_text}".strip())

    return "\n\n".join(texts).strip()


def ocr_image_bytes(image_bytes: bytes, lang: str = "eng") -> str:
    _configure_binaries()
    img = Image.open(BytesIO(image_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return (pytesseract.image_to_string(img, lang=lang) or "").strip()
