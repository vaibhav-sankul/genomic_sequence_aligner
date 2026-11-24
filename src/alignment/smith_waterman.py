"""
Smith-Waterman Dynamic Programming Local Sequence Aligner (Pure Python / NumPy Baseline)
Serves as the biological and computational accuracy ground-truth for GPU validation.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .traceback import TracebackEngine, AlignmentResult
from .alignment_utils import encode_sequence


class SmithWatermanAligner:
    """
    Standard implementation of the Smith-Waterman local alignment algorithm.
    Uses dynamic programming with configurable match, mismatch, and linear gap penalties.
    """

    def __init__(self, match_score: int = 2, mismatch_score: int = -1, gap_penalty: int = -2):
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_penalty = gap_penalty
        self.traceback_engine = TracebackEngine(
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_penalty=gap_penalty
        )

    def compute_matrix(self, query_seq: str, ref_seq: str) -> Tuple[np.ndarray, Tuple[int, int], int]:
        """
        Fills the (len(query)+1, len(ref)+1) dynamic programming score matrix.
        Returns:
            (H_matrix, (max_i, max_j), max_score)
        """
        n = len(query_seq)
        m = len(ref_seq)

        H = np.zeros((n + 1, m + 1), dtype=np.int32)
        max_score = 0
        max_pos = (0, 0)

        for i in range(1, n + 1):
            q_char = query_seq[i - 1]
            for j in range(1, m + 1):
                r_char = ref_seq[j - 1]
                match_val = self.match_score if q_char == r_char else self.mismatch_score

                diag = H[i - 1, j - 1] + match_val
                up = H[i - 1, j] + self.gap_penalty
                left = H[i, j - 1] + self.gap_penalty

                score = max(0, diag, up, left)
                H[i, j] = score

                if score > max_score:
                    max_score = score
                    max_pos = (i, j)

        return H, max_pos, max_score

    def align(
        self,
        query_seq: str,
        ref_seq: str,
        query_id: str = "query",
        ref_id: str = "reference",
        qualities: Optional[List[int]] = None
    ) -> AlignmentResult:
        """
        Executes local alignment between a single query and a reference sequence.
        Returns complete AlignmentResult including traceback, coordinates, and CIGAR.
        """
        if not query_seq or not ref_seq:
            return self.traceback_engine.backtrack(
                np.zeros((1, 1), dtype=np.int32), "", "", 0, 0, query_id, ref_id
            )

        H, (max_i, max_j), _ = self.compute_matrix(query_seq, ref_seq)
        return self.traceback_engine.backtrack(
            H=H,
            query_seq=query_seq,
            ref_seq=ref_seq,
            max_i=max_i,
            max_j=max_j,
            query_id=query_id,
            ref_id=ref_id,
            qualities=qualities
        )

    def align_batch(
        self,
        reads: List[Dict[str, Any]],
        ref_seq: str,
        ref_id: str = "ref"
    ) -> List[AlignmentResult]:
        """
        Sequentially aligns a batch of read dictionaries against the reference sequence.
        Useful for reference baseline benchmarking.
        """
        results: List[AlignmentResult] = []
        for read in reads:
            res = self.align(
                query_seq=read["sequence"],
                ref_seq=ref_seq,
                query_id=read["id"],
                ref_id=ref_id,
                qualities=read.get("quality")
            )
            results.append(res)
        return results
