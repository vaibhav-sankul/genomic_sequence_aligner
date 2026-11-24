"""
Sequence Indexing and Burrows-Wheeler Transform (BWT) Module
Provides exact-match seed finding and candidate region selection to prune alignment search spaces.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


class BWTIndex:
    """
    Educational Burrows-Wheeler Transform (BWT) with LF-mapping for seed finding.
    
    Architectural Note:
    -------------------
    This implementation computes the standard BWT via suffix array sorting and
    supports backward search for exact k-mer seeds. In production aligners like BWA-MEM:
    1. A compressed 2-bit FM-index is used with checkpointed Occ tables (sampled every 64/128 bases).
    2. Suffix array positions are sampled (e.g. every 32 bases) to reduce memory from O(N log N) to O(N).
    3. Bidirectional BWT is applied for maximal exact matches (MEMs).
    This module provides the exact theoretical BWT principles in clean Python.
    """

    def __init__(self, reference_seq: str):
        self.ref_seq = reference_seq.upper()
        self.sentinel = "$"
        self.text = self.ref_seq + self.sentinel
        self.n = len(self.text)
        
        # Build Suffix Array and BWT
        self.suffix_array = self._build_suffix_array()
        self.bwt = "".join(self.text[(sa - 1) % self.n] for sa in self.suffix_array)
        
        # Build First-column counts and character tally tables
        self.c_table, self.occ_table = self._build_lf_tables()

    def _build_suffix_array(self) -> List[int]:
        """Builds suffix array by sorting suffix start indices."""
        suffixes = sorted(range(self.n), key=lambda i: self.text[i:])
        return suffixes

    def _build_lf_tables(self) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
        """Builds C count table and Occurrence tally table for backward search."""
        counts: Dict[str, int] = {}
        for char in self.bwt:
            counts[char] = counts.get(char, 0) + 1

        sorted_chars = sorted(counts.keys())
        c_table: Dict[str, int] = {}
        total = 0
        for ch in sorted_chars:
            c_table[ch] = total
            total += counts[ch]

        # Build Occurrence table Occ(c, k): count of char c in bwt[0:k]
        occ_table: Dict[str, List[int]] = {ch: [0] * (self.n + 1) for ch in counts}
        current_counts = {ch: 0 for ch in counts}

        for i, ch in enumerate(self.bwt):
            current_counts[ch] += 1
            for c in counts:
                occ_table[c][i + 1] = current_counts[c]

        return c_table, occ_table

    def backward_search(self, pattern: str) -> List[int]:
        """
        Locates all exact occurrences of pattern in the reference using BWT backward search.
        Returns 0-based start positions in reference sequence.
        """
        pattern = pattern.upper()
        if not pattern or any(ch not in self.c_table for ch in pattern):
            return []

        # Start with interval spanning the entire BWT
        k = len(pattern) - 1
        char = pattern[k]
        if char not in self.c_table:
            return []

        sp = self.c_table[char]
        char_count = self.occ_table[char][-1]
        ep = sp + char_count - 1

        while sp <= ep and k > 0:
            k -= 1
            char = pattern[k]
            if char not in self.c_table:
                return []

            sp = self.c_table[char] + self.occ_table[char][sp]
            ep = self.c_table[char] + self.occ_table[char][ep + 1] - 1

        if sp <= ep:
            return sorted([self.suffix_array[i] for i in range(sp, ep + 1)])
        return []

    def get_candidate_windows(self, read_seq: str, seed_len: int = 15, padding: int = 50) -> List[Tuple[int, int]]:
        """
        Extracts candidate reference coordinate windows (start, end) for a read
        using seeds extracted from the query.
        """
        if len(read_seq) < seed_len:
            return [(0, len(self.ref_seq))]

        seed = read_seq[:seed_len]
        hits = self.backward_search(seed)

        if not hits:
            # Fall back to middle seed
            mid = len(read_seq) // 2
            seed = read_seq[mid:mid + seed_len]
            hits = self.backward_search(seed)

        if not hits:
            return [(0, len(self.ref_seq))]

        windows = []
        for hit in hits:
            w_start = max(0, hit - padding)
            w_end = min(len(self.ref_seq), hit + len(read_seq) + padding)
            windows.append((w_start, w_end))
        return windows


class SequenceKmerIndex:
    """
    Hash-based k-mer index for ultra-fast candidate region lookup.
    """

    def __init__(self, reference_seq: str, k: int = 12):
        self.ref_seq = reference_seq.upper()
        self.k = k
        self.index: Dict[str, List[int]] = {}
        self._build_index()

    def _build_index(self) -> None:
        for i in range(len(self.ref_seq) - self.k + 1):
            kmer = self.ref_seq[i:i + self.k]
            if 'N' not in kmer:
                if kmer not in self.index:
                    self.index[kmer] = []
                self.index[kmer].append(i)

    def find_seeds(self, query_seq: str) -> List[int]:
        """Finds candidate reference offsets matching query k-mers."""
        hits = []
        for i in range(0, len(query_seq) - self.k + 1, self.k):
            kmer = query_seq[i:i + self.k]
            if kmer in self.index:
                hits.extend(self.index[kmer])
        return sorted(list(set(hits)))
