from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    chunk_id: int
    text: str
    start_char: int
    end_char: int


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[TextChunk]:
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[TextChunk] = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_str = text[start:end].strip()

        if chunk_text_str:
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text_str,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_id += 1

        
        if end == len(text):
            break
        start = end - overlap

    return chunks
