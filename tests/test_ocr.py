from io import BytesIO

from PIL import Image

from app.ingest import ocr
from app.main import _is_probably_scanned_pdf


def _png_bytes() -> bytes:
    image = Image.new("RGB", (20, 20), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_scanned_pdf_detection_ignores_page_markers():
    marker_only_text = "\n\n".join(f"[PAGE {i}]" for i in range(1, 20))

    assert _is_probably_scanned_pdf(marker_only_text)
    assert not _is_probably_scanned_pdf("[PAGE 1]\nThis page contains enough extracted text to skip OCR.")


def test_ocr_image_bytes_falls_back_to_tesseract(monkeypatch):
    monkeypatch.setattr(ocr, "_gemini_ocr", lambda _: (_ for _ in ()).throw(RuntimeError("no gemini")))
    monkeypatch.setattr(ocr.pytesseract, "image_to_string", lambda *_args, **_kwargs: "fallback text")

    assert ocr.ocr_image_bytes(_png_bytes()) == "fallback text"


def test_ocr_pdf_file_uses_gemini_for_converted_pages(monkeypatch, tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    page_image = Image.new("RGB", (20, 20), color="white")

    monkeypatch.setattr(ocr, "convert_from_path", lambda *_args, **_kwargs: [page_image])
    monkeypatch.setattr(ocr, "_gemini_ocr", lambda _image_bytes: "invoice total 100")

    assert ocr.ocr_pdf_file(pdf_path, max_pages=1) == "[PAGE 1]\ninvoice total 100"
