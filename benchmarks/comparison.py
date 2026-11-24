"""
Comparative Analysis and Algorithmic Scope Breakdown
Analyzes differences between full local alignment (Smith-Waterman) and seed-and-extend heuristics (BWA/Bowtie2).
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_algorithmic_comparison():
    """Outputs clear comparison of algorithmic complexity and biological scope."""
    table = Table(title="🔬 Algorithmic Scope & Hardware Comparison", header_style="bold magenta", box=None)
    table.add_column("Dimension", style="bold cyan")
    table.add_column("Smith-Waterman (Our Aligner)", style="white")
    table.add_column("BWA-MEM / Bowtie2", style="white")

    table.add_row(
        "Algorithm Class",
        "Exact Local Dynamic Programming",
        "Heuristic Seed-and-Extend (FM-Index)"
    )
    table.add_row(
        "Computational Complexity",
        "O(N × M) per read (Compute-bound)",
        "O(N) expected (Memory/Cache-bound)"
    )
    table.add_row(
        "Optimal Sensitivity",
        "100% Mathematically Optimal Local Alignment",
        "Heuristic; may miss highly divergent regions"
    )
    table.add_row(
        "GPU Acceleration",
        "Massively parallelized DP grid on CUDA cores",
        "Predominantly CPU-optimized tree traversals"
    )
    table.add_row(
        "Best Application",
        "High-sensitivity local realignment & variant discovery",
        "Ultra-high throughput whole genome mapping"
    )

    console.print(Panel(table, title="[bold]Algorithmic Scope Distinction[/bold]", border_style="cyan"))


if __name__ == "__main__":
    print_algorithmic_comparison()
