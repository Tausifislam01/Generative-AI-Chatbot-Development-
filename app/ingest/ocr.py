from __future__ import annotations
import os
import base64
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

def _gemini_ocr(image_bytes: bytes) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")
        
    img_b64 = base64.b64encode(image_bytes).decode()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Extract all text from this image exactly as written. Do not add any extra commentary or formatting. If there is no text, return nothing."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
    )
    res = llm.invoke([msg])
    return str(res.content).strip()

def ocr_image_file(file_path: Path, lang: str = "eng") -> str:
    _configure_binaries()
    try:
        with open(file_path, "rb") as f:
            return _gemini_ocr(f.read())
    except Exception:
        pass

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
        page_text = ""
        try:
            buf = BytesIO()
            page_img.save(buf, format="JPEG")
            page_text = _gemini_ocr(buf.getvalue())
        except Exception:
            page_text = pytesseract.image_to_string(page_img, lang=lang) or ""
            
        texts.append(f"[PAGE {i}]\n{page_text}".strip())

    return "\n\n".join(texts).strip()

def ocr_image_bytes(image_bytes: bytes, lang: str = "eng") -> str:
    _configure_binaries()
    try:
        return _gemini_ocr(image_bytes)
    except Exception:
        pass

    img = Image.open(BytesIO(image_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return (pytesseract.image_to_string(img, lang=lang) or "").strip()
