"""Smith-Waterman sequence alignment module."""
from .smith_waterman import SmithWatermanAligner
from .traceback import TracebackEngine, AlignmentResult
from .alignment_utils import encode_sequence, decode_sequence

__all__ = ["SmithWatermanAligner", "TracebackEngine", "AlignmentResult", "encode_sequence", "decode_sequence"]
