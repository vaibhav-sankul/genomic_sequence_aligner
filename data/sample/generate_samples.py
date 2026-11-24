"""
Utility script to generate realistic sample reference and sequencing reads (FASTA & FASTQ).
"""

import random
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent

# Set deterministic seed for repeatable testing
random.seed(42)

BASES = ['A', 'C', 'G', 'T']


def generate_reference(length: int = 1500) -> str:
    """Generates a synthetic reference genome sequence."""
    return "".join(random.choices(BASES, weights=[0.28, 0.22, 0.22, 0.28], k=length))


def mutate_read(seq: str, snp_rate: float = 0.02, indel_rate: float = 0.01) -> str:
    """Introduces biological variations (SNPs, insertions, deletions) into a read."""
    mutated = []
    for base in seq:
        r = random.random()
        if r < snp_rate:
            # SNP substitution
            mutated.append(random.choice([b for b in BASES if b != base]))
        elif r < snp_rate + indel_rate:
            # Deletion (skip base)
            continue
        elif r < snp_rate + (2 * indel_rate):
            # Insertion
            mutated.append(base)
            mutated.append(random.choice(BASES))
        else:
            mutated.append(base)
    return "".join(mutated)


def create_sample_dataset():
    ref_seq = generate_reference(1500)
    ref_file = SAMPLE_DIR / "reference.fasta"
    with open(ref_file, "w") as f:
        f.write(">chr1 Synthetic Reference Chromosome 1 (1500bp)\n")
        for i in range(0, len(ref_seq), 80):
            f.write(ref_seq[i:i+80] + "\n")

    num_reads = 100
    read_len = 80

    fasta_file = SAMPLE_DIR / "sample.fasta"
    fastq_file = SAMPLE_DIR / "sample.fastq"

    with open(fasta_file, "w") as fa, open(fastq_file, "w") as fq:
        for i in range(num_reads):
            start = random.randint(0, len(ref_seq) - read_len - 10)
            orig_read = ref_seq[start:start + read_len]
            mut_read = mutate_read(orig_read)
            
            read_id = f"read_{i+1:04d}_pos_{start}"
            
            # Write FASTA
            fa.write(f">{read_id} Synthetic read from pos {start}\n")
            fa.write(f"{mut_read}\n")

            # Write FASTQ (Phred+33 ASCII quality strings, typical Q30-Q38)
            fq.write(f"@{read_id}\n")
            fq.write(f"{mut_read}\n")
            fq.write("+\n")
            quals = "".join(chr(random.randint(30, 40) + 33) for _ in mut_read)
            fq.write(f"{quals}\n")

    print(f"Generated sample datasets successfully:")
    print(f"  Reference: {ref_file} ({len(ref_seq)} bp)")
    print(f"  FASTA:     {fasta_file} ({num_reads} reads)")
    print(f"  FASTQ:     {fastq_file} ({num_reads} reads)")


if __name__ == "__main__":
    create_sample_dataset()
