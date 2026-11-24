"""
CSV Alignment Report Generator
Outputs detailed tabular alignment metrics (scores, coordinates, CIGAR, identity %).
"""

import csv
from pathlib import Path
from typing import List
from ..alignment.traceback import AlignmentResult


class CSVReportGenerator:
    """
    Exports comprehensive per-read alignment details to CSV format.
    """

    @staticmethod
    def write_csv(alignments: List[AlignmentResult], output_path: str | Path) -> Path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "query_id", "ref_id", "score", "ref_start", "ref_end",
            "query_start", "query_end", "query_aligned", "ref_aligned",
            "cigar", "matches", "mismatches", "gaps", "identity_pct"
        ]

        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for aln in alignments:
                writer.writerow(aln.to_dict())

        return out_file
