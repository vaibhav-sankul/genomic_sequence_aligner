"""
CUDA-Accelerated Genomic Aligner with Seamless CPU Fallback
Coordinates device memory transfers, kernel execution, and traceback reconstruction.
"""

import math
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from .cuda_detector import CUDADetector, HardwareInfo
from .memory_manager import DeviceMemoryManager
from .gpu_kernels import cuda_batch_sw_kernel, cpu_batch_sw_kernel
from ..alignment.alignment_utils import encode_sequence, pad_sequences
from ..alignment.traceback import TracebackEngine, AlignmentResult
from ..alignment.smith_waterman import SmithWatermanAligner

logger = logging.getLogger(__name__)


class CUDAAligner:
    """
    High-throughput sequence aligner utilizing CUDA hardware acceleration.
    Automatically and transparently falls back to optimized parallel CPU execution
    if CUDA hardware or runtimes are unavailable.
    """

    def __init__(
        self,
        match_score: int = 2,
        mismatch_score: int = -1,
        gap_penalty: int = -2,
        threads_per_block: int = 128,
        device_mode: str = "auto",
        use_pinned_memory: bool = True
    ):
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_penalty = gap_penalty
        self.threads_per_block = threads_per_block
        self.device_mode = device_mode.lower()
        
        # Hardware inspection
        self.hw_info: HardwareInfo = CUDADetector.get_hardware_info()
        self.use_cuda = self._determine_execution_mode()

        # Components
        self.mem_manager = DeviceMemoryManager(use_pinned_memory=use_pinned_memory)
        self.traceback_engine = TracebackEngine(
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_penalty=gap_penalty
        )
        self.cpu_baseline_aligner = SmithWatermanAligner(
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_penalty=gap_penalty
        )

        logger.info(f"Initialized CUDAAligner [Mode: {'CUDA' if self.use_cuda else 'CPU_FALLBACK'}]")

    def _determine_execution_mode(self) -> bool:
        if self.device_mode == "cpu":
            return False
        if self.device_mode == "cuda":
            if not self.hw_info.has_cuda:
                logger.warning("CUDA mode requested but no CUDA device available. Falling back to CPU.")
                return False
            return True
        # "auto"
        return self.hw_info.has_cuda

    def align_batch(
        self,
        reads: List[Dict[str, Any]],
        ref_seq: str,
        ref_id: str = "chr1"
    ) -> List[AlignmentResult]:
        """
        Aligns a batch of genomic reads against the reference sequence.
        Uses CUDA GPU kernels if active, otherwise parallel CPU kernel.
        """
        batch_size = len(reads)
        if batch_size == 0 or not ref_seq:
            return []

        # Encode reference
        ref_arr = encode_sequence(ref_seq)
        ref_len = len(ref_arr)

        # Encode queries and compute padded batch
        query_arrays = [encode_sequence(r["sequence"]) for r in reads]
        padded_queries, q_lengths = pad_sequences(query_arrays, pad_val=5)

        # Prepare score output buffers
        scores_out = np.zeros(batch_size, dtype=np.int32)
        max_i_out = np.zeros(batch_size, dtype=np.int32)
        max_j_out = np.zeros(batch_size, dtype=np.int32)

        if self.use_cuda and cuda_batch_sw_kernel is not None:
            # ----------------------------------------------------
            # CUDA HARDWARE ACCELERATION PATH
            # ----------------------------------------------------
            try:
                # Device allocations via Memory Manager
                d_queries = self.mem_manager.host_to_device(padded_queries)
                d_q_lens = self.mem_manager.host_to_device(q_lengths)
                d_ref = self.mem_manager.host_to_device(ref_arr)
                d_scores = self.mem_manager.host_to_device(scores_out)
                d_max_i = self.mem_manager.host_to_device(max_i_out)
                d_max_j = self.mem_manager.host_to_device(max_j_out)

                blocks_per_grid = math.ceil(batch_size / self.threads_per_block)

                # Launch CUDA Kernel
                cuda_batch_sw_kernel[blocks_per_grid, self.threads_per_block](
                    d_queries,
                    d_q_lens,
                    d_ref,
                    ref_len,
                    self.match_score,
                    self.mismatch_score,
                    self.gap_penalty,
                    d_scores,
                    d_max_i,
                    d_max_j
                )
                self.mem_manager.synchronize()

                # Transfer results back from device
                self.mem_manager.device_to_host(d_scores, scores_out)
                self.mem_manager.device_to_host(d_max_i, max_i_out)
                self.mem_manager.device_to_host(d_max_j, max_j_out)

            except Exception as e:
                logger.error(f"CUDA execution failed: {e}. Executing CPU fallback.")
                cpu_batch_sw_kernel(
                    padded_queries, q_lengths, ref_arr, ref_len,
                    self.match_score, self.mismatch_score, self.gap_penalty,
                    scores_out, max_i_out, max_j_out
                )
        else:
            # ----------------------------------------------------
            # HIGH-PERFORMANCE PARALLEL CPU FALLBACK PATH
            # ----------------------------------------------------
            cpu_batch_sw_kernel(
                padded_queries, q_lengths, ref_arr, ref_len,
                self.match_score, self.mismatch_score, self.gap_penalty,
                scores_out, max_i_out, max_j_out
            )

        # --------------------------------------------------------
        # RECONSTRUCT ACCURATE TRACEBACK FOR EACH READ
        # --------------------------------------------------------
        results: List[AlignmentResult] = []
        for idx, read in enumerate(reads):
            q_seq = read["sequence"]
            score = int(scores_out[idx])
            end_i = int(max_i_out[idx])
            end_j = int(max_j_out[idx])

            if score <= 0 or end_i <= 0 or end_j <= 0:
                # Unaligned or zero score
                results.append(AlignmentResult(
                    query_id=read["id"],
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
                    identity_pct=0.0,
                    quality_scores=read.get("quality")
                ))
            else:
                # Perform windowed dynamic programming traceback to construct CIGAR and metrics
                # To maximize efficiency, we bound the matrix to the aligned region
                window_start_j = max(0, end_j - (len(q_seq) + 30))
                window_ref = ref_seq[window_start_j:end_j]
                
                # Align in window
                res = self.cpu_baseline_aligner.align(
                    query_seq=q_seq[:end_i],
                    ref_seq=window_ref,
                    query_id=read["id"],
                    ref_id=ref_id,
                    qualities=read.get("quality")
                )
                # Adjust reference coordinates back to global coordinate frame
                res.ref_start += window_start_j
                res.ref_end += window_start_j
                results.append(res)

        return results
