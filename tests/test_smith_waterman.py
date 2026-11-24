"""
Unit tests for Smith-Waterman local alignment and traceback engine.
"""

import pytest
from src.alignment.smith_waterman import SmithWatermanAligner
from src.alignment.alignment_utils import encode_sequence, decode_sequence, generate_cigar


def test_smith_waterman_exact_match():
    aligner = SmithWatermanAligner(match_score=2, mismatch_score=-1, gap_penalty=-2)
    seq = "ATGCATGCAT"
    res = aligner.align(seq, seq)

    # 10 bases * 2 = 20 score
    assert res.score == 20
    assert res.query_aligned == seq
    assert res.ref_aligned == seq
    assert res.matches == 10
    assert res.mismatches == 0
    assert res.gaps == 0
    assert res.identity_pct == 100.0
    assert res.cigar == "10M"


def test_smith_waterman_with_mismatch():
    aligner = SmithWatermanAligner(match_score=2, mismatch_score=-1, gap_penalty=-2)
    # One mismatch: 'T' replaced by 'C' at position 2
    query = "AGTCAGTC"
    ref =   "AGCCAGTC"
    res = aligner.align(query, ref)

    # 7 matches (14) - 1 mismatch (1) = 13 score
    assert res.score == 13
    assert res.matches == 7
    assert res.mismatches == 1
    assert res.gaps == 0
    assert res.identity_pct == 87.5
    assert res.cigar == "8M"


def test_smith_waterman_with_gap():
    aligner = SmithWatermanAligner(match_score=2, mismatch_score=-1, gap_penalty=-2)
    # Deletion in query: missing 'G', flanked by matching sequences to force gap extension
    query = "ATCGAAAAAA"
    ref =   "ATCGGAAAAAA"
    res = aligner.align(query, ref)

    # 10 matches (20) - 1 gap (2) = 18
    assert res.score == 18
    assert res.gaps == 1
    assert res.matches == 10
    assert "D" in res.cigar or "I" in res.cigar


def test_alignment_utils():
    seq = "ACGTN"
    enc = encode_sequence(seq)
    assert list(enc) == [0, 1, 2, 3, 4]
    dec = decode_sequence(enc)
    assert dec == seq

    cigar = generate_cigar("AT-CG", "ATGCG")
    assert cigar == "2M1D2M"
