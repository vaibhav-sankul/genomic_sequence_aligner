# High-Throughput Genomic Sequence Aligner (CUDA/Python)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-Numba%20Accelerated-green.svg)](https://numba.pydata.org/)
[![Tests](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-throughput, memory-bounded Next-Generation Sequencing (NGS) alignment pipeline developed with **Python 3.11**, **Numba CUDA**, and multi-core CPU parallel execution.

Built from scratch as a 4-week software development internship project for **Zaalima Development Private Limited**.

---

## 🌟 Key Features

1. **Massively Parallel CUDA Acceleration:** Smith-Waterman local alignment kernel executed across GPU threads using Numba CUDA (`@cuda.jit`).
2. **Transparent CPU Fallback:** Automatic non-crashing fallback to multi-core parallel CPU execution (`@njit(parallel=True)`) when an NVIDIA GPU is not detected.
3. **Memory-Bounded Streaming Ingestion:** Chunk-based FASTA and FASTQ readers that stream gigabyte-scale datasets without memory overflows or out-of-memory (OOM) crashes.
4. **Hardware Memory Management:** Device memory pooling and page-locked host memory (`pinned_array`) to eliminate PCIe allocation bottlenecks.
5. **Multi-Threaded Asynchronous Prefetching:** Overlaps disk I/O, sequence preprocessing, and compute kernels using `concurrent.futures.ThreadPoolExecutor`.
6. **Burrows-Wheeler Transform (BWT) Indexing:** Suffix array construction and backward search for exact k-mer seed finding and candidate region selection.
7. **Standards-Compliant Reporting:** Produces standard **VCF 4.2** variant call format files (SNPs, insertions, deletions) and comprehensive tabular CSV reports.
8. **Real-Time Terminal Dashboard:** Powered by `Rich`, displaying live throughput (reads/sec), progress bars, memory telemetry, and dynamic ETA.
9. **Benchmarking Suite & Visualizations:** Automated performance benchmarking generating CSV metrics and Matplotlib comparison charts.
10. **Comprehensive Test Suite:** 17 automated Pytest unit tests validating biological accuracy parity, ingestion, memory management, and VCF calling.

---

## 📁 Project Architecture

```
genomic_sequence_aligner/
│
├── main.py                      # Main application entry point & CLI
├── config.py                    # Global parameters and scoring configurations
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation & execution guide
├── .gitignore                   # Git exclusion rules
│
├── data/
│   ├── input/                   # Custom user input files
│   ├── output/                  # Generated CSV and VCF 4.2 reports
│   └── sample/                  # Synthetic reference and sequencing reads
│       ├── reference.fasta      # Synthetic reference genome (1500 bp)
│       ├── sample.fasta         # Sample FASTA reads (100 reads)
│       └── sample.fastq         # Sample FASTQ reads with Phred scores
│
├── src/
│   ├── ingestion/               # FASTA, FASTQ, and chunk streaming readers
│   ├── alignment/               # Smith-Waterman dynamic programming & traceback
│   ├── gpu/                     # CUDA detector, device memory manager & kernels
│   ├── preprocessing/           # Burrows-Wheeler Transform (BWT) & k-mer index
│   ├── processing/              # Multi-threaded prefetching & pipeline orchestration
│   ├── reporting/               # VCF 4.2, CSV reports, and chart visualizer
│   └── dashboard/               # Rich live terminal dashboard
│
├── tests/                       # Automated Pytest suite (17 tests)
│   ├── test_fasta_reader.py
│   ├── test_smith_waterman.py
│   ├── test_cuda.py
│   ├── test_pipeline.py
│   └── test_vcf.py
│
├── benchmarks/                  # Performance benchmarking harness & charts
│   ├── benchmark.py
│   ├── comparison.py
│   ├── benchmark_results.csv
│   └── *.png                    # Generated visual comparison charts
│
└── docs/                        # Complete technical documentation
    ├── architecture.md          # System architecture & data flow
    ├── algorithms.md            # Mathematical formulation & DP equations
    └── project_report.md        # Final internship evaluation report
```

---

## 🚀 Installation & Environment Setup

### 1. Prerequisites
- **Python 3.11** installed on your system.
- *(Optional for GPU acceleration)*: NVIDIA GPU with CUDA Toolkit installed. If unavailable, the pipeline automatically runs in optimized parallel CPU mode.

### 2. Setup Virtual Environment

```powershell
# Navigate to the project directory
cd E:\.gemini\genomic_sequence_aligner

# Create a virtual environment with Python 3.11
py -3.11 -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 💻 Execution & Usage

### 1. Run Complete Pipeline (Default Sample Dataset)

```powershell
python main.py --input data/sample/sample.fasta
```

### 2. Run with FASTQ Sequencing Reads

```powershell
python main.py --input data/sample/sample.fastq
```

### 3. Enforce Specific Compute Target

```powershell
# Force CUDA execution (warns and falls back if unavailable)
python main.py --input data/sample/sample.fasta --device cuda

# Force CPU multi-core execution
python main.py --input data/sample/sample.fasta --device cpu
```

### 4. Custom Chunk Size and Worker Threads

```powershell
python main.py --input data/sample/sample.fasta --chunk-size 500 --workers 4
```

### 5. Run Performance Benchmarks

```powershell
python main.py --benchmark
```

### 6. Display Algorithmic Comparison

```powershell
python main.py --compare
```

---

## 🧪 Running Automated Tests

Run the complete test suite with verbose output:

```powershell
pytest tests/ -v
```

All 17 tests validate:
- FASTA/FASTQ parsing, sanitization, and Phred quality scores
- Memory-mapped chunk indexing and RAM thresholding
- Smith-Waterman exact matches, mismatches, and indel gaps
- 100% biological accuracy parity between CPU baseline and accelerated aligner
- Burrows-Wheeler Transform suffix sorting and backward search
- VCF 4.2 single nucleotide polymorphisms (SNPs) and indel anchoring

---

## 📊 Benchmark Summary

On a standard reference benchmark (100 reads against a 1,500 bp reference sequence):

| Engine | Execution Mode | Wall Time | Throughput | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Python SW** | Sequential CPU | 6.40s | 15.6 reads/s | 1.0x (Baseline) |
| **Vectorized Parallel SW** | Parallel CPU Fallback | 0.52s | 193.9 reads/s | **12.42x Speedup** |
| **CUDA Accelerated SW** | NVIDIA GPU (Grid Kernel) | Fast DMA | Hardware-Bound | **Massive Parallelism** |

---

## 📄 Output Formats

### 1. CSV Alignment Report (`data/output/alignments.csv`)
Columns: `query_id`, `ref_id`, `score`, `ref_start`, `ref_end`, `query_start`, `query_end`, `query_aligned`, `ref_aligned`, `cigar`, `matches`, `mismatches`, `gaps`, `identity_pct`.

### 2. VCF 4.2 Variant Report (`data/output/variants.vcf`)
Standard format containing:
- Reference genome sequence and contig lengths
- 1-based variant coordinate (`POS`)
- Reference (`REF`) and Alternate (`ALT`) alleles for SNPs and anchored indels
- Phred-scaled variant quality score (`QUAL`)
- Detailed annotations: Read Depth (`DP`), Allele Count (`AC`), Allele Frequency (`AF`), and Variant Type (`TYPE`).

---

## 📋 Weekly Milestones Completed

- [x] **Week 1:** FASTA/FASTQ chunked ingestion; memory-mapped file reading; pure Python Smith-Waterman baseline.
- [x] **Week 2:** Numba CUDA dynamic programming kernel; GPU memory manager & pooling; CPU-GPU accuracy parity testing.
- [x] **Week 3:** Asynchronous `ThreadPoolExecutor` prefetching; Burrows-Wheeler Transform (BWT) indexing; live `Rich` terminal dashboard.
- [x] **Week 4:** Multi-engine performance benchmarking with Matplotlib charts; automated VCF 4.2 variant reporting; full documentation suite.
