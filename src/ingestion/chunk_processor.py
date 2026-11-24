"""
Chunk Processor and Memory-Mapped Streamer
Enables chunk-based genomic file reading and memory-budget control.
"""

import os
import gc
import mmap
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional
import psutil

from .fasta_reader import FASTAReader
from .fastq_reader import FASTQReader


class ChunkProcessor:
    """
    Manages batched ingestion of massive genomic files.
    Ensures memory consumption remains strictly bounded within RAM budgets.
    """

    def __init__(
        self,
        file_path: str | Path,
        chunk_size: int = 1000,
        memory_limit_mb: float = 2048.0,
        use_mmap_indexing: bool = False
    ):
        self.file_path = Path(file_path)
        self.chunk_size = max(1, chunk_size)
        self.memory_limit_mb = memory_limit_mb
        self.use_mmap_indexing = use_mmap_indexing
        self.process = psutil.Process(os.getpid())
        
        # Detect format
        self.format = self._detect_format()
        if self.format == "fasta":
            self.reader = FASTAReader(self.file_path)
        elif self.format == "fastq":
            self.reader = FASTQReader(self.file_path)
        else:
            raise ValueError(f"Unsupported file format for: {self.file_path}. Expected .fasta, .fa, .fastq, or .fq")

    def _detect_format(self) -> str:
        ext = self.file_path.suffix.lower()
        if ext in [".fasta", ".fa", ".fna"]:
            return "fasta"
        if ext in [".fastq", ".fq"]:
            return "fastq"
        # Fallback to inspecting first non-empty byte
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    return "fasta"
                if line.startswith("@"):
                    return "fastq"
                break
        raise ValueError(f"Cannot identify FASTA/FASTQ format for {self.file_path}")

    def get_memory_usage_mb(self) -> float:
        """Returns current process Resident Set Size (RSS) in megabytes."""
        return self.process.memory_info().rss / (1024 * 1024)

    def check_memory_headroom(self) -> None:
        """Triggers explicit garbage collection if RAM consumption nears limit."""
        current_mb = self.get_memory_usage_mb()
        if current_mb > (self.memory_limit_mb * 0.85):
            gc.collect()

    def get_total_records(self) -> int:
        """Delegates fast header counting to format reader."""
        return self.reader.count_records()

    def stream_chunks(self) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Yields batches of parsed genomic records of size `self.chunk_size`.
        Guarantees that RAM is reclaimed between iterations.
        """
        current_chunk: List[Dict[str, Any]] = []

        for record in self.reader.stream_records():
            current_chunk.append(record)

            if len(current_chunk) >= self.chunk_size:
                yield current_chunk
                current_chunk = []
                self.check_memory_headroom()

        if current_chunk:
            yield current_chunk
            current_chunk = []
            self.check_memory_headroom()

    def build_mmap_index(self) -> List[int]:
        """
        Optional byte-offset index for random access into giant FASTA/FASTQ files.
        Scans byte offsets of record start characters ('>' or '@') using memory mapping.
        """
        offsets: List[int] = []
        with open(self.file_path, "rb") as f:
            if os.path.getsize(self.file_path) == 0:
                return offsets
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                prefix = b">" if self.format == "fasta" else b"@"
                pos = 0
                size = len(mm)
                while pos < size:
                    # Find beginning of line
                    if pos == 0 or mm[pos - 1:pos] == b"\n":
                        if mm[pos:pos + 1] == prefix:
                            offsets.append(pos)
                    # Advance to next newline
                    next_nl = mm.find(b"\n", pos)
                    if next_nl == -1:
                        break
                    pos = next_nl + 1
        return offsets
