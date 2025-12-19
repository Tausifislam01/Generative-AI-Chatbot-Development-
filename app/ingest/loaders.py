from pathlib import Path


def load_txt(file_path: Path) -> str:
    
    return file_path.read_text(encoding="utf-8", errors="ignore")
