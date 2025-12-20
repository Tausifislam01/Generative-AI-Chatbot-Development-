from pathlib import Path
from typing import List
from pypdf import PdfReader


def load_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages_text: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        pages_text.append(t)
    return "\n".join(pages_text).strip()
