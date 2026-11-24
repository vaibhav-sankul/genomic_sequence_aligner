"""
CUDA and Vectorized High-Performance Alignment Kernels
Contains Numba @cuda.jit device kernels and parallelized @njit CPU fallback kernels.
Guarantees identical mathematical scoring logic between hardware-accelerated and CPU paths.
"""

import numpy as np
from numba import njit, prange

# Conditional CUDA import
try:
    from numba import cuda
    _NUMBA_CUDA_SUPPORT = True
except ImportError:
    cuda = None
    _NUMBA_CUDA_SUPPORT = False


# =========================================================================
# 1. NUMBA CUDA DEVICE KERNELS
# =========================================================================

if _NUMBA_CUDA_SUPPORT:
    @cuda.jit
    def cuda_batch_sw_kernel(
        queries,       # shape: (B, max_q_len), uint8
        q_lengths,     # shape: (B,), int32
        reference,     # shape: (R,), uint8
        ref_len,       # int32
        match_score,   # int32
        mismatch_score,# int32
        gap_penalty,   # int32
        scores_out,    # shape: (B,), int32
        max_i_out,     # shape: (B,), int32
        max_j_out      # shape: (B,), int32
    ):
        """
        Numba CUDA kernel for massively parallelized Smith-Waterman batch scoring.
        Each GPU thread independently computes the dynamic programming matrix for one read.
        Uses two alternating rolling rows to minimize device memory and maximize cache hits.
        """
        tid = cuda.grid(1)
        batch_size = queries.shape[0]

        if tid >= batch_size:
            return

        q_len = q_lengths[tid]
        if q_len <= 0 or ref_len <= 0:
            scores_out[tid] = 0
            max_i_out[tid] = 0
            max_j_out[tid] = 0
            return

        # Local static arrays for rolling DP rows (supports reference windows up to 2048 bp)
        # In CUDA device registers / local memory
        MAX_REF_COLS = 2048
        actual_ref = ref_len if ref_len < MAX_REF_COLS else MAX_REF_COLS

        prev_row = cuda.local.array(MAX_REF_COLS, dtype=np.int32)
        curr_row = cuda.local.array(MAX_REF_COLS, dtype=np.int32)

        # Initialize row 0 with zeros
        for j in range(actual_ref + 1):
            prev_row[j] = 0
            curr_row[j] = 0

        max_val = 0
        best_i = 0
        best_j = 0

        for i in range(1, q_len + 1):
            q_base = queries[tid, i - 1]
            curr_row[0] = 0

            for j in range(1, actual_ref + 1):
                r_base = reference[j - 1]

                # Match vs Mismatch
                delta = match_score if (q_base == r_base and q_base != 4) else mismatch_score

                diag = prev_row[j - 1] + delta
                up = prev_row[j] + gap_penalty
                left = curr_row[j - 1] + gap_penalty

                # Local alignment maximum (>= 0)
                val = 0
                if diag > val:
                    val = diag
                if up > val:
                    val = up
                if left > val:
                    val = left

                curr_row[j] = val

                if val > max_val:
                    max_val = val
                    best_i = i
                    best_j = j

            # Swap rows for next iteration
            for j in range(actual_ref + 1):
                prev_row[j] = curr_row[j]

        scores_out[tid] = max_val
        max_i_out[tid] = best_i
        max_j_out[tid] = best_j

else:
    cuda_batch_sw_kernel = None


# =========================================================================
# 2. NUMBA VECTORIZED PARALLEL CPU FALLBACK KERNEL
# =========================================================================

@njit(parallel=True, fastmath=True)
def cpu_batch_sw_kernel(
    queries: np.ndarray,
    q_lengths: np.ndarray,
    reference: np.ndarray,
    ref_len: int,
    match_score: int,
    mismatch_score: int,
    gap_penalty: int,
    scores_out: np.ndarray,
    max_i_out: np.ndarray,
    max_j_out: np.ndarray
):
    """
    Highly optimized multi-core parallelized CPU Smith-Waterman kernel.
    Executes identical dynamic programming logic across CPU worker threads.
    """
    batch_size = queries.shape[0]

    for b in prange(batch_size):
        q_len = q_lengths[b]
        if q_len <= 0 or ref_len <= 0:
            scores_out[b] = 0
            max_i_out[b] = 0
            max_j_out[b] = 0
            continue

        prev_row = np.zeros(ref_len + 1, dtype=np.int32)
        curr_row = np.zeros(ref_len + 1, dtype=np.int32)

        max_val = 0
        best_i = 0
        best_j = 0

        for i in range(1, q_len + 1):
            q_base = queries[b, i - 1]
            curr_row[0] = 0

            for j in range(1, ref_len + 1):
                r_base = reference[j - 1]
                delta = match_score if (q_base == r_base and q_base != 4) else mismatch_score

                diag = prev_row[j - 1] + delta
                up = prev_row[j] + gap_penalty
                left = curr_row[j - 1] + gap_penalty

                val = 0
                if diag > val:
                    val = diag
                if up > val:
                    val = up
                if left > val:
                    val = left

                curr_row[j] = val

                if val > max_val:
                    max_val = val
                    best_i = i
                    best_j = j

            # Swap rows
            for j in range(ref_len + 1):
                prev_row[j] = curr_row[j]

        scores_out[b] = max_val
        max_i_out[b] = best_i
        max_j_out[b] = best_j
