"""
Traceback Engine and Alignment Results Data Structure
Handles backtracking along optimal dynamic programming paths and computes alignment metrics.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

from .alignment_utils import generate_cigar


@dataclass
class AlignmentResult:
    """Encapsulates full alignment metrics for a single query read aligned to reference."""
    query_id: str
    ref_id: str
    score: int
    query_aligned: str
    ref_aligned: str
    query_start: int
    query_end: int
    ref_start: int
    ref_end: int
    cigar: str = ""
    matches: int = 0
    mismatches: int = 0
    gaps: int = 0
    identity_pct: float = 0.0
    quality_scores: Optional[List[int]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "ref_id": self.ref_id,
            "score": self.score,
            "query_aligned": self.query_aligned,
            "ref_aligned": self.ref_aligned,
            "query_start": self.query_start,
            "query_end": self.query_end,
            "ref_start": self.ref_start,
            "ref_end": self.ref_end,
            "cigar": self.cigar,
            "matches": self.matches,
            "mismatches": self.mismatches,
            "gaps": self.gaps,
            "identity_pct": self.identity_pct
        }


class TracebackEngine:
    """
    Standard Smith-Waterman traceback processor.
    Reconstructs the optimal local alignment path from the peak score cell down to 0.
    """

    def __init__(self, match_score: int = 2, mismatch_score: int = -1, gap_penalty: int = -2):
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_penalty = gap_penalty

    def backtrack(
        self,
        H: np.ndarray,
        query_seq: str,
        ref_seq: str,
        max_i: int,
        max_j: int,
        query_id: str = "query",
        ref_id: str = "reference",
        qualities: Optional[List[int]] = None
    ) -> AlignmentResult:
        """
        Backtracks through dynamic programming matrix H from cell (max_i, max_j).
        H has shape (len(query) + 1, len(ref) + 1).
        """
        max_score = int(H[max_i, max_j])
        if max_score <= 0:
            return AlignmentResult(
                query_id=query_id,
                ref_id=ref_id,
                score=0,
                query_aligned="",
                ref_aligned="",
                query_start=0,
                query_end=0,
                ref_start=0,
                ref_end=0,
                cigar="",
                matches=0,
                mismatches=0,
                gaps=0,
                identity_pct=0.0
            )

        aligned_q: List[str] = []
        aligned_r: List[str] = []

        curr_i = max_i
        curr_j = max_j

        query_end = curr_i
        ref_end = curr_j

        while curr_i > 0 and curr_j > 0 and H[curr_i, curr_j] > 0:
            current_val = H[curr_i, curr_j]
            q_char = query_seq[curr_i - 1]
            r_char = ref_seq[curr_j - 1]

            match_delta = self.match_score if q_char == r_char else self.mismatch_score

            # Check diagonal move
            if current_val == H[curr_i - 1, curr_j - 1] + match_delta:
                aligned_q.append(q_char)
                aligned_r.append(r_char)
                curr_i -= 1
                curr_j -= 1
            # Check up move (gap in reference / insertion in query)
            elif current_val == H[curr_i - 1, curr_j] + self.gap_penalty:
                aligned_q.append(q_char)
                aligned_r.append('-')
                curr_i -= 1
            # Check left move (gap in query / deletion from reference)
            elif current_val == H[curr_i, curr_j - 1] + self.gap_penalty:
                aligned_q.append('-')
                aligned_r.append(r_char)
                curr_j -= 1
            else:
                # Fallback to diagonal if non-affine boundary condition
                aligned_q.append(q_char)
                aligned_r.append(r_char)
                curr_i -= 1
                curr_j -= 1

        query_start = curr_i
        ref_start = curr_j

        aligned_query_str = "".join(reversed(aligned_q))
        aligned_ref_str = "".join(reversed(aligned_r))

        # Metrics calculation
        matches = 0
        mismatches = 0
        gaps = 0

        for qc, rc in zip(aligned_query_str, aligned_ref_str):
            if qc == '-' or rc == '-':
                gaps += 1
            elif qc == rc:
                matches += 1
            else:
                mismatches += 1

        total_positions = len(aligned_query_str)
        identity = (matches / total_positions * 100.0) if total_positions > 0 else 0.0
        cigar_str = generate_cigar(aligned_query_str, aligned_ref_str)

        return AlignmentResult(
            query_id=query_id,
            ref_id=ref_id,
            score=max_score,
            query_aligned=aligned_query_str,
            ref_aligned=aligned_ref_str,
            query_start=query_start,
            query_end=query_end,
            ref_start=ref_start,
            ref_end=ref_end,
            cigar=cigar_str,
            matches=matches,
            mismatches=mismatches,
            gaps=gaps,
            identity_pct=round(identity, 2),
            quality_scores=qualities
        )
