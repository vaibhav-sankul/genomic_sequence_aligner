"""
Variant Call Format (VCF 4.2) Report Generator
Extracts substitutions (SNPs), insertions, and deletions from local alignment results
and formats them into standardized VCF 4.2 specifications.
"""

import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from ..alignment.traceback import AlignmentResult


class VCFGenerator:
    """
    Generates standards-compliant VCF 4.2 variant files from alignment outputs.
    Aggregates allele evidence across reads to distinguish true variants from sequencing errors.
    """

    def __init__(
        self,
        reference_seq: str,
        contig_id: str = "chr1",
        min_depth: int = 1,
        min_allele_freq: float = 0.1,
        min_quality: float = 20.0
    ):
        self.ref_seq = reference_seq
        self.contig_id = contig_id
        self.min_depth = min_depth
        self.min_allele_freq = min_allele_freq
        self.min_quality = min_quality

    def _extract_variants_from_alignment(self, res: AlignmentResult) -> List[Dict[str, Any]]:
        """
        Walks aligned query and reference strings to identify differences.
        Reference coordinates are 1-based standard VCF.
        """
        variants: List[Dict[str, Any]] = []
        if not res.query_aligned or not res.ref_aligned:
            return variants

        ref_pos = res.ref_start  # 0-based start
        q_pos = 0

        for qc, rc in zip(res.query_aligned, res.ref_aligned):
            if qc != '-' and rc != '-':
                # Substitution / Match
                if qc != rc:
                    # Single Nucleotide Polymorphism (SNP)
                    vcf_pos = ref_pos + 1  # 1-based
                    variants.append({
                        "pos": vcf_pos,
                        "type": "SNP",
                        "ref": rc,
                        "alt": qc,
                        "read_id": res.query_id
                    })
                ref_pos += 1
                q_pos += 1
            elif qc == '-':
                # Deletion from reference
                vcf_pos = ref_pos + 1
                variants.append({
                    "pos": vcf_pos,
                    "type": "DEL",
                    "ref": rc,
                    "alt": "-",
                    "read_id": res.query_id
                })
                ref_pos += 1
            elif rc == '-':
                # Insertion into query
                vcf_pos = ref_pos  # anchored at preceding base
                variants.append({
                    "pos": vcf_pos,
                    "type": "INS",
                    "ref": "-",
                    "alt": qc,
                    "read_id": res.query_id
                })
                q_pos += 1

        return variants

    def call_variants(self, alignments: List[AlignmentResult]) -> List[Dict[str, Any]]:
        """
        Aggregates raw alignment differences across all reads to call supported variants.
        """
        # Group by (pos, type, ref, alt)
        locus_counts: Dict[Tuple[int, str, str, str], int] = defaultdict(int)
        locus_depth: Dict[int, int] = defaultdict(int)

        # Track depth across covered reference positions
        for aln in alignments:
            if aln.score <= 0:
                continue
            for p in range(aln.ref_start + 1, aln.ref_end + 1):
                locus_depth[p] += 1

            for var in self._extract_variants_from_alignment(aln):
                key = (var["pos"], var["type"], var["ref"], var["alt"])
                locus_counts[key] += 1

        called: List[Dict[str, Any]] = []

        for (pos, var_type, ref, alt), alt_count in sorted(locus_counts.items(), key=lambda x: x[0][0]):
            depth = max(alt_count, locus_depth.get(pos, alt_count))
            af = alt_count / depth if depth > 0 else 0.0

            if depth >= self.min_depth and af >= self.min_allele_freq:
                # Format REF and ALT alleles for VCF indel specification
                if var_type == "DEL":
                    anchor_idx = max(0, pos - 2)
                    anchor_base = self.ref_seq[anchor_idx] if anchor_idx < len(self.ref_seq) else "N"
                    vcf_ref = anchor_base + ref
                    vcf_alt = anchor_base
                    actual_pos = max(1, pos - 1)
                elif var_type == "INS":
                    anchor_idx = max(0, pos - 1)
                    anchor_base = self.ref_seq[anchor_idx] if anchor_idx < len(self.ref_seq) else "N"
                    vcf_ref = anchor_base
                    vcf_alt = anchor_base + alt
                    actual_pos = max(1, pos)
                else:  # SNP
                    vcf_ref = ref
                    vcf_alt = alt
                    actual_pos = pos

                # Estimate phred-scaled quality: -10 * log10(1 - AF)
                qual = min(60.0, max(20.0, round(af * 50.0 + 10.0, 1)))

                called.append({
                    "chrom": self.contig_id,
                    "pos": actual_pos,
                    "id": f"var_{actual_pos}_{vcf_ref}_{vcf_alt}",
                    "ref": vcf_ref,
                    "alt": vcf_alt,
                    "qual": qual,
                    "filter": "PASS",
                    "dp": depth,
                    "ac": alt_count,
                    "af": round(af, 3),
                    "type": var_type
                })

        return called

    def write_vcf(self, alignments: List[AlignmentResult], output_path: str | Path) -> Path:
        """
        Calls variants and writes a standardized VCF 4.2 file.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        variants = self.call_variants(alignments)

        timestamp = time.strftime("%Y%m%d")

        with open(out_file, "w", encoding="utf-8") as f:
            # VCF 4.2 Headers
            f.write("##fileformat=VCFv4.2\n")
            f.write(f"##fileDate={timestamp}\n")
            f.write("##source=HighThroughputGenomicSequenceAligner_v1.0\n")
            f.write(f"##reference={self.contig_id}\n")
            f.write(f"##contig=<ID={self.contig_id},length={len(self.ref_seq)}>\n")
            f.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total read depth at locus">\n')
            f.write('##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes">\n')
            f.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n')
            f.write('##INFO=<ID=TYPE,Number=A,Type=String,Description="Variant type: SNP, INS, DEL">\n')
            f.write('##FILTER=<ID=PASS,Description="All filters passed">\n')
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

            for v in variants:
                info_field = f"DP={v['dp']};AC={v['ac']};AF={v['af']};TYPE={v['type']}"
                f.write(f"{v['chrom']}\t{v['pos']}\t{v['id']}\t{v['ref']}\t{v['alt']}\t{v['qual']}\t{v['filter']}\t{info_field}\n")

        return out_file
