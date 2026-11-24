"""
Unit tests for VCF 4.2 variant report generation.
"""

import pytest
from pathlib import Path
from src.alignment.traceback import AlignmentResult
from src.reporting.vcf_generator import VCFGenerator


def test_vcf_generator_snp(tmp_path):
    ref_seq = "ACGTACGTACGT"
    vcf_gen = VCFGenerator(reference_seq=ref_seq, contig_id="chr1", min_depth=1, min_allele_freq=0.1)

    # Simulated alignment with a single SNP at position 3 (0-based idx 2: G -> C)
    aln = AlignmentResult(
        query_id="read_01",
        ref_id="chr1",
        score=20,
        query_aligned="ACCTACGT",
        ref_aligned=  "ACGTACGT",
        query_start=0,
        query_end=8,
        ref_start=0,
        ref_end=8,
        cigar="8M",
        matches=7,
        mismatches=1,
        gaps=0,
        identity_pct=87.5
    )

    variants = vcf_gen.call_variants([aln])
    assert len(variants) == 1
    v = variants[0]
    assert v["pos"] == 3  # 1-based coordinate
    assert v["ref"] == "G"
    assert v["alt"] == "C"
    assert v["type"] == "SNP"
    assert v["filter"] == "PASS"

    # Test file writing
    out_vcf = tmp_path / "test.vcf"
    written_path = vcf_gen.write_vcf([aln], out_vcf)
    assert written_path.exists()

    with open(written_path, "r") as f:
        lines = f.readlines()
        assert any(l.startswith("##fileformat=VCFv4.2") for l in lines)
        assert any("chr1\t3\t" in l and "\tG\tC\t" in l for l in lines)


def test_vcf_generator_indel(tmp_path):
    ref_seq = "ACGTACGT"
    vcf_gen = VCFGenerator(reference_seq=ref_seq, contig_id="chr1", min_depth=1, min_allele_freq=0.1)

    # Deletion in query (gap in query, ref has 'G')
    aln_del = AlignmentResult(
        query_id="read_del",
        ref_id="chr1",
        score=15,
        query_aligned="AC-TACGT",
        ref_aligned=  "ACGTACGT",
        query_start=0,
        query_end=7,
        ref_start=0,
        ref_end=8,
        cigar="2M1D5M",
        matches=7,
        mismatches=0,
        gaps=1,
        identity_pct=87.5
    )

    variants = vcf_gen.call_variants([aln_del])
    assert len(variants) == 1
    v = variants[0]
    assert v["type"] == "DEL"
