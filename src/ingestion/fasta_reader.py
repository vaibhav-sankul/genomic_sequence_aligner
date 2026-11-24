"""
FASTA File Ingestion Module
Provides memory-efficient streaming and validation for FASTA genomic files.
"""

import os
from pathlib import Path
from typing import Generator, Optional, List, Dict, Any
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


class FASTAReader:
    """
    Incremental FASTA stream reader.
    Streams records one-by-one or in batches without loading entire files into memory.
    """

    VALID_DNA = set("ACGTNacgtn")

    def __init__(self, file_path: str | Path, validate_nucleotides: bool = True):
        self.file_path = Path(file_path)
        self.validate_nucleotides = validate_nucleotides
        if not self.file_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.file_path}")

    @property
    def file_size_bytes(self) -> int:
        return self.file_path.stat().st_size

    def stream_records(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yields parsed FASTA records as lightweight dictionaries.
        Avoids accumulating large BioPython SeqRecord objects in RAM.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    seq_str = str(record.seq).upper()
                    
                    if self.validate_nucleotides:
                        # Clean/sanitize any unexpected characters by mapping to 'N'
                        sanitized = "".join(ch if ch in "ACGT" else "N" for ch in seq_str)
                    else:
                        sanitized = seq_str

                    yield {
                        "id": record.id,
                        "description": record.description,
                        "sequence": sanitized,
                        "length": len(sanitized),
                        "quality": None  # FASTA has no quality scores
                    }
        except Exception as e:
            raise RuntimeError(f"Error reading FASTA file {self.file_path}: {str(e)}") from e

    def count_records(self) -> int:
        """
        Counts total records in the FASTA file by fast line-scanning of headers.
        Extremely fast and requires negligible memory.
        """
        count = 0
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith(">"):
                    count += 1
        return count
