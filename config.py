"""
High-Throughput Genomic Sequence Aligner (CUDA/Python)
Global Configuration File
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
SAMPLE_DIR = DATA_DIR / "sample"
DOCS_DIR = BASE_DIR / "docs"
BENCHMARK_DIR = BASE_DIR / "benchmarks"

# Ensure runtime directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AlignmentConfig:
    """Smith-Waterman scoring configuration as specified."""
    match_score: int = 2
    mismatch_score: int = -1
    gap_penalty: int = -2
    min_score_threshold: int = 10  # Minimum alignment score to report


@dataclass
class HardwareConfig:
    """Hardware acceleration and resource allocation parameters."""
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    threads_per_block: int = 128
    gpu_batch_size: int = 2048
    use_pinned_memory: bool = True
    cpu_workers: int = field(default_factory=lambda: max(1, os.cpu_count() or 4))
    prefetch_batches: int = 2


@dataclass
class IngestionConfig:
    """Genomic file ingestion and memory-efficient streaming parameters."""
    chunk_size: int = 1000  # Number of sequences per chunk
    max_sequence_length: int = 1000  # Truncate or warn if exceeds GPU kernel limits
    use_mmap: bool = True  # Enable memory-mapped file indexing
    quality_threshold: int = 20  # Minimum average Phred quality for FASTQ filtering


@dataclass
class ReportConfig:
    """Output generation and reporting configurations."""
    generate_vcf: bool = True
    generate_csv: bool = True
    vcf_version: str = "VCFv4.2"
    reference_name: str = "chr1"
    output_vcf_name: str = "variants.vcf"
    output_csv_name: str = "alignments.csv"
    benchmark_report_name: str = "benchmark_results.csv"


# Default application configuration instance
DEFAULT_ALIGNMENT_CONFIG = AlignmentConfig()
DEFAULT_HARDWARE_CONFIG = HardwareConfig()
DEFAULT_INGESTION_CONFIG = IngestionConfig()
DEFAULT_REPORT_CONFIG = ReportConfig()
