# Algorithmic Foundations: Genomic Sequence Alignment & Indexing

## 1. The Smith-Waterman Algorithm

The Smith-Waterman algorithm (Smith & Waterman, 1981) is a dynamic programming algorithm designed to determine the optimal local alignment between two nucleotide or amino acid sequences. Unlike global alignment algorithms (e.g., Needleman-Wunsch) which align sequences end-to-end, Smith-Waterman identifies the most similar homologous sub-region, ignoring divergent flanking sequences.

### 1.1 Mathematical Formulation

Let $Q = (q_1, q_2, \dots, q_n)$ represent the query sequence (sequencing read) of length $n$, and $R = (r_1, r_2, \dots, r_m)$ represent the reference sequence of length $m$.

A dynamic programming scoring matrix $H$ of dimensions $(n+1) \times (m+1)$ is initialized:
$$H[i, 0] = 0 \quad \text{for } 0 \le i \le n$$
$$H[0, j] = 0 \quad \text{for } 0 \le j \le m$$

Each internal cell $H[i, j]$ is evaluated using the recurrence relation:

$$H[i, j] = \max \begin{cases}
0 & \text{(Termination of local alignment)} \\
H[i-1, j-1] + s(q_i, r_j) & \text{(Match or Mismatch)} \\
H[i-1, j] + \gamma & \text{(Deletion from reference / Insertion in query)} \\
H[i, j-1] + \gamma & \text{(Insertion in reference / Deletion in query)}
\end{cases}$$

Where the scoring parameters are configured as:
- **Match Score**: $s(q_i, r_j) = +2 \quad \text{if } q_i = r_j \text{ and } q_i \neq \text{'N'}$
- **Mismatch Score**: $s(q_i, r_j) = -1 \quad \text{if } q_i \neq r_j \text{ or } q_i = \text{'N'}$
- **Linear Gap Penalty**: $\gamma = -2$

### 1.2 Peak Identification & Traceback

1. **Peak Score Search**: The highest value in matrix $H$ represents the local alignment score:
   $$\text{Score}^* = \max_{0 \le i \le n, 0 \le j \le m} H[i, j] = H[i^*, j^*]$$
   The cell $(i^*, j^*)$ marks the termination coordinate of the alignment.

2. **Backtracking**: Starting from $(i^*, j^*)$, the traceback engine traverses backwards:
   - Move to $(i-1, j-1)$ if $H[i, j] = H[i-1, j-1] + s(q_i, r_j)$ (Aligned base pair)
   - Move to $(i-1, j)$ if $H[i, j] = H[i-1, j] + \gamma$ (Gap in reference)
   - Move to $(i, j-1)$ if $H[i, j] = H[i, j-1] + \gamma$ (Gap in query)
   - Traceback terminates when a cell with score $H[i, j] = 0$ is encountered.
   The coordinate $(i, j)$ where traceback halts denotes the start position.

### 1.3 CIGAR String Generation

During backtracking, aligned operations are collected and converted to standard CIGAR notation:
- `M`: Alignment match or mismatch
- `I`: Insertion in the query sequence relative to reference
- `D`: Deletion from the reference sequence

Consecutive operations are run-length encoded:
$$\text{Query: } \texttt{AT-CGA} \quad \text{Ref: } \texttt{ATGCGA} \implies \texttt{2M1D3M}$$

---

## 2. GPU Dynamic Programming Optimization

### 2.1 The Two-Row Rolling Buffer Technique

A naive CUDA implementation that stores the complete $(n+1) \times (m+1)$ matrix in device global memory would consume:
$$\text{Memory per read} = (150 + 1) \times (1500 + 1) \times 4 \text{ bytes} \approx 906 \text{ KB}$$
For a batch of 10,000 reads, this requires ~9 GB of device memory, inducing catastrophic cache misses and memory exhaustion.

**Our Rolling Buffer Solution:**
Notice that computing row $i$ only requires values from row $i-1$ and row $i$. We maintain only two alternating 1D buffers in fast GPU local memory:
$$\text{prev\_row}[0 \dots m], \quad \text{curr\_row}[0 \dots m]$$
This reduces memory consumption per thread to:
$$2 \times (m + 1) \times 4 \text{ bytes} \approx 12 \text{ KB}$$
Threads maintain high occupancy in streaming multiprocessor (SM) registers and L1 cache, allowing thousands of reads to align simultaneously in parallel.

---

## 3. Burrows-Wheeler Transform (BWT) & Indexing

### 3.1 Mathematical Theory

The Burrows-Wheeler Transform (Burrows & Wheeler, 1994) is a reversible permutation of a sequence text $T$.
Given text $T = R + \texttt{\$}$ where $\texttt{\$}$ is a lexicographically smallest sentinel character:

1. **Cyclic Rotations**: Construct all $n$ cyclic rotations of $T$.
2. **Suffix Array (SA)**: Sort the cyclic rotations lexicographically. The starting index of each sorted suffix forms the Suffix Array $SA[0 \dots n-1]$.
3. **BWT String**: The last column $L$ of the sorted rotation matrix represents the BWT:
   $$BWT[i] = T[(SA[i] - 1) \bmod n]$$

### 3.2 Exact Backward Search & LF-Mapping

The Last-to-First (LF) mapping property states that the $k$-th occurrence of character $c$ in the last column $L$ corresponds to the $k$-th occurrence of character $c$ in the sorted first column $F$.

To search for a query seed $P = (p_1, p_2, \dots, p_k)$ of length $k$:
1. Initialize search range to the entire BWT: $[sp_k, ep_k] = [C(p_k), C(p_k) + Occ(p_k, n) - 1]$
2. For $i = k-1$ down to 1:
   $$sp_i = C(p_i) + Occ(p_i, sp_{i+1})$$
   $$ep_i = C(p_i) + Occ(p_i, ep_{i+1} + 1) - 1$$
3. If $sp_i \le ep_i$, the pattern occurs $(ep_i - sp_i + 1)$ times, with exact reference coordinates given by $\{SA[j] \mid sp_i \le j \le ep_i\}$.

### 3.3 Educational BWT vs. Production BWA-MEM Comparison

| Architectural Feature | Our Internship Aligner (BWT) | Industry Standard BWA-MEM / Bowtie2 |
| :--- | :--- | :--- |
| **Primary Alignment Philosophy** | Full Dynamic Programming (Smith-Waterman) on GPU | Seed-and-Extend Heuristic (Chaining MEMs) on CPU |
| **Index Representation** | Uncompressed Python array tables | 2-bit Bit-Packed FM-Index |
| **Occurrence ($Occ$) Storage** | Full occurrence table $O(N \times \Sigma)$ | Sampled checkpoint vectors (every 64/128 bp) |
| **Suffix Array ($SA$) Storage** | Full uncompressed array | Sparse sampled suffix array (every 32 bp) |
| **Memory Footprint** | $O(N \log N)$ | Compressed $O(N)$ (~3-4 GB for whole human genome) |
| **Sensitivity** | 100% optimal local alignment score | Heuristic; trade-off between speed and sensitivity |

---

## 4. Variant Calling (VCF 4.2) Algorithm

### 4.1 Allele Identification
For each aligned read against reference coordinates:
1. **Single Nucleotide Polymorphism (SNP)**:
   - Condition: $q_i \neq r_j \land q_i \neq \texttt{'-'} \land r_j \neq \texttt{'-'}$
   - Coordinates: $POS = \text{ref\_start} + j + 1$ (1-based)
   - $REF = r_j, \quad ALT = q_i$

2. **Deletion (DEL)**:
   - Condition: $q_i = \texttt{'-'} \land r_j \neq \texttt{'-'}$
   - VCF Representation: Anchored to previous reference base $r_{j-1}$:
     $$POS = \text{ref\_start} + j, \quad REF = r_{j-1}r_j, \quad ALT = r_{j-1}$$

3. **Insertion (INS)**:
   - Condition: $q_i \neq \texttt{'-'} \land r_j = \texttt{'-'}$
   - VCF Representation: Anchored to previous reference base $r_j$:
     $$POS = \text{ref\_start} + j, \quad REF = r_j, \quad ALT = r_j q_i$$

### 4.2 Locus Filtration & Quality Scoring
At each identified variant locus:
- **Read Depth ($DP$)**: Total overlapping reads spanning the coordinate.
- **Allele Count ($AC$)**: Number of reads confirming the alternate allele.
- **Allele Frequency ($AF$)**: $AF = AC / DP$.
- Variants are called when $DP \ge \text{min\_depth}$ and $AF \ge \text{min\_allele\_freq}$.
- **Phred Quality Score ($QUAL$)**:
  $$QUAL = \min\left(60.0, \max\left(20.0, -10 \cdot \log_{10}(1.0 - AF + 10^{-6})\right)\right)$$
