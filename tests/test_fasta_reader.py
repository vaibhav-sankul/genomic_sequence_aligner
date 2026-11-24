"""
Unit tests for FASTA, FASTQ, and ChunkProcessor Ingestion modules.
"""

import pytest
from pathlib import Path
import config
from src.ingestion.fasta_reader import FASTAReader
from src.ingestion.fastq_reader import FASTQReader
from src.ingestion.chunk_processor import ChunkProcessor

SAMPLE_DIR = config.SAMPLE_DIR


def test_fasta_reader():
    fasta_path = SAMPLE_DIR / "sample.fasta"
    assert fasta_path.exists(), "Sample FASTA file does not exist"
    
    reader = FASTAReader(fasta_path)
    count = reader.count_records()
    assert count == 100, f"Expected 100 records, got {count}"

    records = list(reader.stream_records())
    assert len(records) == 100
    first = records[0]
    assert "id" in first
    assert "sequence" in first
    assert len(first["sequence"]) > 0
    assert all(c in "ACGTN" for c in first["sequence"])


def test_fastq_reader():
    fastq_path = SAMPLE_DIR / "sample.fastq"
    assert fastq_path.exists(), "Sample FASTQ file does not exist"

    reader = FASTQReader(fastq_path)
    count = reader.count_records()
    assert count == 100, f"Expected 100 records, got {count}"

    records = list(reader.stream_records())
    assert len(records) == 100
    first = records[0]
    assert "id" in first
    assert "sequence" in first
    assert "quality" in first
    assert len(first["quality"]) == len(first["sequence"])
    assert first["avg_quality"] >= 25


def test_chunk_processor():
    fasta_path = SAMPLE_DIR / "sample.fasta"
    chunk_proc = ChunkProcessor(fasta_path, chunk_size=25)
    
    assert chunk_proc.format == "fasta"
    assert chunk_proc.get_total_records() == 100

    chunks = list(chunk_proc.stream_chunks())
    assert len(chunks) == 4  # 100 / 25 = 4 chunks
    assert all(len(c) == 25 for c in chunks)
    assert chunk_proc.get_memory_usage_mb() > 0


def test_chunk_processor_mmap():
    fasta_path = SAMPLE_DIR / "sample.fasta"
    chunk_proc = ChunkProcessor(fasta_path, chunk_size=25, use_mmap_indexing=True)
    offsets = chunk_proc.build_mmap_index()
    assert len(offsets) == 100
