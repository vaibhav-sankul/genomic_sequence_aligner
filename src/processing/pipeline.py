"""
End-to-End Genomic Alignment Pipeline Orchestrator
Coordinates file ingestion, index preparation, multi-threaded alignment, and reporting.
"""

import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..ingestion.chunk_processor import ChunkProcessor
from ..ingestion.fasta_reader import FASTAReader
from ..alignment.traceback import AlignmentResult
from ..gpu.cuda_detector import CUDADetector
from ..gpu.cuda_aligner import CUDAAligner
from ..preprocessing.sequence_index import BWTIndex, SequenceKmerIndex
from ..dashboard.terminal_dashboard import TerminalDashboard
from .parallel_processor import BatchParallelExecutor

logger = logging.getLogger(__name__)


class AlignmentPipeline:
    """
    Main orchestration pipeline executing high-throughput sequence alignment.
    Manages streaming input batches, candidate indexing, GPU/CPU execution, and progress tracking.
    """

    def __init__(
        self,
        reference_path: str | Path,
        device_mode: str = "auto",
        chunk_size: int = 1000,
        num_workers: int = 4,
        match_score: int = 2,
        mismatch_score: int = -1,
        gap_penalty: int = -2,
        use_bwt: bool = False,
        enable_dashboard: bool = True
    ):
        self.reference_path = Path(reference_path)
        self.device_mode = device_mode
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        self.use_bwt = use_bwt
        self.enable_dashboard = enable_dashboard

        # Load reference genome
        if not self.reference_path.exists():
            raise FileNotFoundError(f"Reference sequence file not found: {self.reference_path}")
        
        ref_reader = FASTAReader(self.reference_path)
        ref_records = list(ref_reader.stream_records())
        if not ref_records:
            raise ValueError(f"No sequence records found in reference file: {self.reference_path}")
        
        self.reference_record = ref_records[0]
        self.ref_seq = self.reference_record["sequence"]
        self.ref_id = self.reference_record["id"]

        # Hardware inspection
        self.hw_info = CUDADetector.get_hardware_info()

        # Aligner engine
        self.aligner = CUDAAligner(
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_penalty=gap_penalty,
            device_mode=self.device_mode
        )

        # Optional BWT Index
        self.bwt_index: Optional[BWTIndex] = None
        if self.use_bwt:
            logger.info("Constructing Burrows-Wheeler Transform index...")
            self.bwt_index = BWTIndex(self.ref_seq)

    def run(self, input_file: str | Path) -> List[AlignmentResult]:
        """
        Executes the alignment pipeline over the target input FASTA/FASTQ file.
        Returns the collected list of AlignmentResult objects.
        """
        input_path = Path(input_file)
        chunk_proc = ChunkProcessor(input_path, chunk_size=self.chunk_size)
        total_records = chunk_proc.get_total_records()

        logger.info(f"Starting alignment on {input_path.name} ({total_records} reads, chunk_size={self.chunk_size})")

        dashboard: Optional[TerminalDashboard] = None
        if self.enable_dashboard:
            dashboard = TerminalDashboard(
                mode="CUDA" if self.aligner.use_cuda else "CPU",
                cuda_available=self.hw_info.has_cuda,
                device_name=self.hw_info.device_name,
                total_sequences=total_records
            )
            dashboard.start()

        all_results: List[AlignmentResult] = []
        start_time = time.time()

        try:
            executor = BatchParallelExecutor(num_workers=self.num_workers, max_prefetch=2)

            def align_chunk(chunk_reads):
                return self.aligner.align_batch(chunk_reads, self.ref_seq, self.ref_id)

            for chunk_results in executor.process_stream(chunk_proc.stream_chunks(), align_chunk):
                all_results.extend(chunk_results)
                if dashboard:
                    dashboard.update(len(chunk_results))

        finally:
            if dashboard:
                dashboard.stop()

        elapsed = time.time() - start_time
        rps = len(all_results) / max(0.001, elapsed)
        logger.info(f"Alignment completed: {len(all_results)} reads in {elapsed:.2f}s ({rps:.1f} reads/s)")

        return all_results
