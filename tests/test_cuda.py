"""
Unit tests for CUDA detection, GPU memory management, and baseline vs accelerated alignment parity.
"""

import pytest
import numpy as np

from src.gpu.cuda_detector import CUDADetector
from src.gpu.memory_manager import DeviceMemoryManager
from src.gpu.cuda_aligner import CUDAAligner
from src.alignment.smith_waterman import SmithWatermanAligner


def test_cuda_detection():
    hw = CUDADetector.get_hardware_info()
    assert hw is not None
    assert isinstance(hw.has_cuda, bool)
    assert hw.active_mode in ["CUDA", "CPU_FALLBACK"]
    # Verify summary dictionary structure
    summary = hw.summary()
    assert "has_cuda" in summary
    assert "active_mode" in summary


def test_memory_manager():
    mm = DeviceMemoryManager(use_pinned_memory=True)
    arr = mm.allocate_pinned_host((10, 10), np.uint8)
    assert arr.shape == (10, 10)
    assert arr.dtype == np.uint8


def test_aligner_accuracy_parity_with_baseline():
    """
    Rigorously tests that CUDAAligner (running either CUDA or vectorized fallback)
    produces identical alignment scores, matches, mismatches, and coordinates as
    the pure-Python Smith-Waterman baseline.
    """
    ref = "CGTAGCTAGCTAGCTAATCGATCGATCGATCGATCG"
    reads = [
        {"id": "r1", "sequence": "GCTAGCTAGCTA", "quality": None},
        {"id": "r2", "sequence": "TCGATCGATCGA", "quality": None},
        {"id": "r3", "sequence": "GCTAACTAGCTA", "quality": None},  # 1 SNP
    ]

    baseline = SmithWatermanAligner(match_score=2, mismatch_score=-1, gap_penalty=-2)
    accelerated = CUDAAligner(match_score=2, mismatch_score=-1, gap_penalty=-2, device_mode="auto")

    base_results = [baseline.align(r["sequence"], ref, r["id"], "ref") for r in reads]
    acc_results = accelerated.align_batch(reads, ref, "ref")

    assert len(base_results) == len(acc_results)

    for b_res, a_res in zip(base_results, acc_results):
        assert a_res.score == b_res.score, f"Score mismatch: {a_res.score} vs {b_res.score} for {a_res.query_id}"
        assert a_res.matches == b_res.matches, f"Match mismatch for {a_res.query_id}"
        assert a_res.mismatches == b_res.mismatches, f"Mismatch mismatch for {a_res.query_id}"
        assert a_res.ref_end == b_res.ref_end, f"Coordinate mismatch for {a_res.query_id}"
