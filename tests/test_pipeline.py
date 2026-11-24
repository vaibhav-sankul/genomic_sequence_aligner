"""
Unit and integration tests for BWT indexing, parallel executor, and the end-to-end pipeline.
"""

import pytest
from pathlib import Path
import config
from src.preprocessing.sequence_index import BWTIndex, SequenceKmerIndex
from src.processing.parallel_processor import BatchParallelExecutor
from src.processing.pipeline import AlignmentPipeline


def test_bwt_index():
    ref = "BANANA"
    bwt = BWTIndex(ref)
    assert bwt.bwt is not None
    # Test exact pattern searches
    hits = bwt.backward_search("ANA")
    assert sorted(hits) == [1, 3]

    hits_na = bwt.backward_search("NA")
    assert sorted(hits_na) == [2, 4]


def test_kmer_index():
    ref = "ACGTACGTACGT"
    kindex = SequenceKmerIndex(ref, k=4)
    hits = kindex.find_seeds("ACGT")
    assert len(hits) >= 1
    assert 0 in hits


def test_parallel_executor():
    executor = BatchParallelExecutor(num_workers=2, max_prefetch=2)
    sample_chunks = [
        [{"id": "r1", "sequence": "ACGT"}],
        [{"id": "r2", "sequence": "TGCA"}],
    ]

    def mock_align(chunk):
        return [f"aligned_{r['id']}" for r in chunk]

    results = list(executor.process_stream(iter(sample_chunks), mock_align))
    assert len(results) == 2
    assert results[0] == ["aligned_r1"]
    assert results[1] == ["aligned_r2"]


def test_end_to_end_pipeline():
    ref_path = config.SAMPLE_DIR / "reference.fasta"
    input_path = config.SAMPLE_DIR / "sample.fasta"
    
    assert ref_path.exists()
    assert input_path.exists()

    pipeline = AlignmentPipeline(
        reference_path=ref_path,
        device_mode="auto",
        chunk_size=20,
        num_workers=2,
        enable_dashboard=False  # Disable live UI in headless test
    )

    results = pipeline.run(input_path)
    assert len(results) == 100
    assert all(r.score > 0 for r in results)
    assert all(r.identity_pct > 80.0 for r in results)
