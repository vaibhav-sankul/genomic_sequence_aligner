# System Architecture: High-Throughput Genomic Sequence Aligner

## 1. Executive Architectural Overview

The **High-Throughput Genomic Sequence Aligner** is a hybrid CPU-GPU bioinformatics pipeline engineered to process large-scale Next-Generation Sequencing (NGS) datasets with bounded memory consumption and maximum computational throughput.

Traditional bioinformatics libraries (such as pure Python BioPython implementations) suffer severe performance degradation when processing millions of short reads due to Python's interpreted bytecode overhead and $O(N \times M)$ dynamic programming complexity. Our architecture solves this bottleneck by orchestrating low-level hardware acceleration (via **Numba CUDA** device kernels) coupled with an asynchronous, multi-threaded CPU ingestion pipeline.

```
+-----------------------------------------------------------------------------------+
|                                 HOST CPU LAYER                                    |
|                                                                                   |
|  +--------------------+     +---------------------+     +----------------------+  |
|  | FASTA/FASTQ Reader | --> |   Chunk Processor   | --> | ThreadPoolExecutor   |  |
|  | (BioPython Stream) |     | (Memory-Budgeted)   |     | (Batch Prefetching)  |  |
|  +--------------------+     +---------------------+     +----------------------+  |
|                                                                    |              |
|                                                        +-----------+-----------+  |
|                                                        |                       |  |
|                                                        v                       v  |
|                                             +--------------------+   +---------+  |
|                                             |  BWT Seed Indexer  |   | Rich    |  |
|                                             |  (Candidate Scope) |   | Dash    |  |
|                                             +--------------------+   +---------+  |
|                                                        |                          |
+--------------------------------------------------------|--------------------------+
                                                         | (Pinned DMA Transfer)
                                                         v
+-----------------------------------------------------------------------------------+
|                                 DEVICE GPU LAYER                                  |
|                                                                                   |
|            +---------------------------------------------------------+            |
|            |               Device Memory Manager (Pool)              |            |
|            +---------------------------------------------------------+            |
|                                         |                                         |
|                                         v                                         |
|            +---------------------------------------------------------+            |
|            | Numba CUDA Grid Kernel (batch_smith_waterman_kernel)    |            |
|            |   - 1 Thread per Read                                   |            |
|            |   - Rolling Dynamic Programming Buffers                 |            |
|            |   - Massively Parallel Matrix Evaluation                |            |
|            +---------------------------------------------------------+            |
|                                         |                                         |
|                                         v (Scores, max_i, max_j)                  |
+-----------------------------------------|-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             DOWNSTREAM REPORTING LAYER                            |
|                                                                                   |
|  +-----------------------------+               +-------------------------------+  |
|  |    VCF 4.2 Variant Caller   |               |   Tabular CSV Aligner Export  |  |
|  | (SNPs, Insertions, Deletions)               |   (CIGAR, Identity, Scores)   |  |
|  +-----------------------------+               +-------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Component Decomposition

### 2.1 Ingestion Layer (`src/ingestion/`)
- **`FASTAReader`**: Implements incremental generator streaming using Biopython's `SeqIO.parse`. DNA characters are sanitized and validated against `{A, C, G, T, N}`.
- **`FASTQReader`**: Extracts sequence bases alongside per-nucleotide Phred quality scores ($Q \in [0, 93]$), filtering reads below configurable quality thresholds.
- **`ChunkProcessor`**: Enforces strict memory budgets. Instead of loading gigabyte-scale datasets into RAM, data is sliced into batches (e.g. 1,000 to 10,000 reads). Integrates memory-mapped file indexing (`mmap`) to scan byte offsets and triggers garbage collection when memory thresholds are approached.

### 2.2 Preprocessing & Indexing Layer (`src/preprocessing/`)
- **`BWTIndex`**: Implements the Burrows-Wheeler Transform, Suffix Array sorting, First-column counts ($C$-table), and Occurrence tally tables ($Occ$-table). Supports exact-match backward search to find candidate reference windows.
- **`SequenceKmerIndex`**: Provides constant-time $k$-mer hash lookups to pinpoint candidate seeding coordinates, significantly narrowing the search envelope for long reference sequences.

### 2.3 Acceleration & Compute Layer (`src/gpu/`)
- **`CUDADetector`**: Interrogates CUDA runtime and driver presence via `numba.cuda.is_available()`. Probes device names, compute capabilities, and total VRAM.
- **`DeviceMemoryManager`**: Eliminates repeated allocation latency on the PCIe bus by maintaining a device buffer pool (`cuda.device_array`) and leveraging page-locked host memory (`cuda.pinned_array`).
- **`gpu_kernels.py`**:
  - `cuda_batch_sw_kernel`: Dispatches a 1D grid of CUDA threads. Each thread executes local dynamic programming over rolling rows of length up to 2048 bp.
  - `cpu_batch_sw_kernel`: Highly optimized Numba `@njit(parallel=True, fastmath=True)` fallback executing identical scoring matrices across host CPU cores when GPU is absent.
- **`CUDAAligner`**: Orchestrates padded numeric sequence conversion, buffer dispatch, kernel execution, and coordinate reconstruction.

### 2.4 Traceback & Metrics (`src/alignment/`)
- **`TracebackEngine`**: Backtracks from $(max\_i, max\_j)$ to cell score 0. Produces aligned query/reference strings, Compact Idiosyncratic Gapped Alignment Report (CIGAR) strings (e.g. `78M1I2D`), exact mismatch counts, and percent identity.
- **`SmithWatermanAligner`**: Pure Python baseline used as the ground-truth accuracy benchmark for testing and parity verification.

### 2.5 Processing & Concurrency (`src/processing/`)
- **`BatchParallelExecutor`**: Uses Python's `concurrent.futures.ThreadPoolExecutor` to overlap file I/O reading, sequence preprocessing, and alignment kernel execution using a bounded producer-consumer queue.
- **`AlignmentPipeline`**: High-level conductor managing input streaming, hardware selection, and data aggregation.

### 2.6 Downstream Reporting (`src/reporting/`)
- **`VCFGenerator`**: Standards-compliant Variant Call Format (VCF 4.2) emitter. Calls single nucleotide polymorphisms (SNPs), insertions, and deletions with reference anchoring and Phred-scaled quality metrics.
- **`CSVReportGenerator`**: Writes tabular alignment metrics for all processed reads.
- **`BenchmarkVisualizer`**: Generates high-resolution Matplotlib graphs comparing execution time, reads/sec throughput, and speedup ratios.

### 2.7 Interactive Telemetry (`src/dashboard/`)
- **`TerminalDashboard`**: Powered by `Rich`, renders live terminal telemetry: current throughput, total/processed/remaining sequences, host RAM, device mode, elapsed time, and dynamic ETA updates.

---

## 3. Hardware Execution & Fallback Strategy

The aligner guarantees crash-free operation across diverse deployment environments:
1. **NVIDIA GPU Available**: High-throughput parallel execution across CUDA streaming multiprocessors (SMs).
2. **CPU-Only Environment**: Automatic transparent fallback to Numba multi-core CPU parallel execution.
3. **Explicit Device Flags**: Users can enforce `--device cpu` or `--device cuda` via the CLI.
4. **Accuracy Invariance**: Both hardware-accelerated and CPU paths use identical scoring parameters (Match: +2, Mismatch: -1, Gap: -2) and produce byte-identical alignment scores.
