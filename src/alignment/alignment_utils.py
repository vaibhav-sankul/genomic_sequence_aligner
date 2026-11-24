"""
Alignment Utilities
Nucleotide numeric encoding, decoding, CIGAR generation, and sequence padding.
"""

from typing import Tuple, List
import numpy as np

# Nucleotide integer encoding map: A:0, C:1, G:2, T:3, N:4, Gap:5
NUC_TO_INT = {
    'A': 0, 'a': 0,
    'C': 1, 'c': 1,
    'G': 2, 'g': 2,
    'T': 3, 't': 3,
    'N': 4, 'n': 4,
    '-': 5
}

INT_TO_NUC = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N', 5: '-'}


def encode_sequence(seq: str) -> np.ndarray:
    """
    Converts a DNA string into an 8-bit unsigned integer NumPy array.
    Unknown characters are safely mapped to 'N' (4).
    """
    return np.array([NUC_TO_INT.get(ch, 4) for ch in seq], dtype=np.uint8)


def decode_sequence(arr: np.ndarray | List[int]) -> str:
    """Converts a numeric array back to standard IUPAC DNA characters."""
    return "".join(INT_TO_NUC.get(int(x), 'N') for x in arr)


def pad_sequences(sequences: List[np.ndarray], pad_val: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pads a list of variable-length numeric sequences to the maximum length in the batch.
    Returns:
        (padded_matrix [batch_size, max_len], lengths_vector [batch_size])
    """
    batch_size = len(sequences)
    if batch_size == 0:
        return np.empty((0, 0), dtype=np.uint8), np.empty((0,), dtype=np.int32)

    lengths = np.array([len(s) for s in sequences], dtype=np.int32)
    max_len = int(lengths.max())

    padded = np.full((batch_size, max_len), fill_value=pad_val, dtype=np.uint8)
    for i, s in enumerate(sequences):
        padded[i, :len(s)] = s

    return padded, lengths


def generate_cigar(aligned_query: str, aligned_ref: str) -> str:
    """
    Computes standard Compact Idiosyncratic Gapped Alignment Report (CIGAR) string.
    Operations:
      'M' = Alignment match or mismatch (sequence alignment match)
      'I' = Insertion in the query (gap in reference)
      'D' = Deletion from reference (gap in query)
    """
    if not aligned_query or not aligned_ref or len(aligned_query) != len(aligned_ref):
        return ""

    ops: List[Tuple[str, int]] = []
    
    for q_char, r_char in zip(aligned_query, aligned_ref):
        if q_char == '-':
            current_op = 'D'
        elif r_char == '-':
            current_op = 'I'
        else:
            current_op = 'M'

        if ops and ops[-1][0] == current_op:
            ops[-1] = (current_op, ops[-1][1] + 1)
        else:
            ops.append((current_op, 1))

    return "".join(f"{count}{op}" for op, count in ops)
