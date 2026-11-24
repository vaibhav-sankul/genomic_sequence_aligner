"""
FASTQ File Ingestion Module
Provides streaming, quality score extraction, and validation for FASTQ sequencing reads.
"""

from pathlib import Path
from typing import Generator, Dict, Any, List, Optional
from Bio import SeqIO


class FASTQReader:
    """
    Incremental FASTQ stream reader.
    Extracts sequences along with Phred quality scores while conserving memory.
    """

    def __init__(self, file_path: str | Path, min_avg_quality: Optional[float] = None):
        self.file_path = Path(file_path)
        self.min_avg_quality = min_avg_quality
        if not self.file_path.exists():
            raise FileNotFoundError(f"FASTQ file not found: {self.file_path}")

    @property
    def file_size_bytes(self) -> int:
        return self.file_path.stat().st_size

    def stream_records(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yields parsed FASTQ records as dictionaries with Phred quality scores.
        Filters out low-quality reads if min_avg_quality is configured.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as handle:
                for record in SeqIO.parse(handle, "fastq"):
                    seq_str = str(record.seq).upper()
                    sanitized = "".join(ch if ch in "ACGT" else "N" for ch in seq_str)
                    
                    # Extract Phred quality scores (typically Q0 - Q40+)
                    qualities = record.letter_annotations.get("phred_quality", [])
                    avg_qual = sum(qualities) / len(qualities) if qualities else 0.0

                    if self.min_avg_quality is not None and avg_qual < self.min_avg_quality:
                        continue

                    yield {
                        "id": record.id,
                        "description": record.description,
                        "sequence": sanitized,
                        "length": len(sanitized),
                        "quality": qualities,
                        "avg_quality": round(avg_qual, 2)
                    }
        except Exception as e:
            raise RuntimeError(f"Error reading FASTQ file {self.file_path}: {str(e)}") from e

    def count_records(self) -> int:
        """
        Fast line-count estimation for FASTQ (4 lines per record standard format).
        """
        line_count = 0
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                line_count += 1
        return line_count // 4
