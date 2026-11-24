"""Data ingestion module for FASTA/FASTQ genomic records."""
from .fasta_reader import FASTAReader
from .fastq_reader import FASTQReader
from .chunk_processor import ChunkProcessor

__all__ = ["FASTAReader", "FASTQReader", "ChunkProcessor"]
