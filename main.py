"""
High-Throughput Genomic Sequence Aligner (CUDA/Python)
Main Entry Point & Command-Line Interface (CLI)

Supports both autonomous CLI execution with rich flags and interactive guided workflow.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

import config
from src.gpu.cuda_detector import CUDADetector
from src.processing.pipeline import AlignmentPipeline
from src.reporting.vcf_generator import VCFGenerator
from src.reporting.csv_report import CSVReportGenerator
from benchmarks.benchmark import run_benchmark
from benchmarks.comparison import print_algorithmic_comparison

console = Console()


def display_welcome_banner(hw_info):
    """STEP 1: Display the project welcome screen."""
    welcome_text = (
        "[bold cyan]🧬 HIGH-THROUGHPUT GENOMIC SEQUENCE ALIGNER[/bold cyan]\n"
        "[bold white]GPU-Accelerated Smith-Waterman Pipeline with Numba CUDA & CPU Fallback[/bold white]\n"
        "[dim]Zaalima Development Internship Final Project[/dim]\n\n"
        f"[bold]Detected Hardware:[/bold] {hw_info.device_name}\n"
        f"[bold]CUDA Acceleration:[/bold] {'[green]AVAILABLE & READY[/green]' if hw_info.has_cuda else '[yellow]UNAVAILABLE (Automatic Multi-Core CPU Fallback)[/yellow]'}\n"
        f"[bold]Execution Mode:[/bold] {hw_info.active_mode}"
    )
    console.print(Panel(welcome_text, title="[bold magenta]SYSTEM INITIALIZATION[/bold magenta]", border_style="cyan"))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-Throughput Genomic Sequence Aligner (CUDA/Python)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to input FASTA or FASTQ sequencing reads file")
    parser.add_argument("--reference", "-r", type=str, default=str(config.SAMPLE_DIR / "reference.fasta"),
                        help="Path to reference genome FASTA file")
    parser.add_argument("--device", "-d", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Execution compute device: 'auto' (preferred), 'cuda', or 'cpu'")
    parser.add_argument("--chunk-size", "-c", type=int, default=1000,
                        help="Number of genomic sequence reads per ingestion chunk")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of host CPU worker threads for I/O prefetching")
    parser.add_argument("--report", action="store_true", default=True,
                        help="Generate comprehensive CSV and VCF variant reports")
    parser.add_argument("--benchmark", "-b", action="store_true",
                        help="Execute multi-engine performance benchmarking")
    parser.add_argument("--compare", action="store_true",
                        help="Display algorithmic scope and hardware comparison table")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Disable rich interactive live dashboard (for headless logs)")
    return parser.parse_args()


def interactive_prompt(args: argparse.Namespace) -> argparse.Namespace:
    """Provides guided user selection if inputs are missing."""
    console.print("\n[bold yellow]No input dataset provided on CLI. Entering Interactive Mode...[/bold yellow]")
    
    # 1. Device selection
    dev_choice = Prompt.ask(
        "Select compute target",
        choices=["auto", "cuda", "cpu"],
        default="auto"
    )
    args.device = dev_choice

    # 2. Input file path
    default_input = str(config.SAMPLE_DIR / "sample.fasta")
    in_file = Prompt.ask(
        "Enter path to query sequence file (FASTA/FASTQ)",
        default=default_input
    )
    args.input = in_file

    # 3. Chunk size
    chunk_str = Prompt.ask("Enter chunk size (reads per batch)", default="1000")
    args.chunk_size = int(chunk_str)

    return args


def main():
    args = parse_arguments()

    # STEP 2: Hardware inspection
    hw_info = CUDADetector.get_hardware_info()
    display_welcome_banner(hw_info)

    # If --compare requested
    if args.compare:
        print_algorithmic_comparison()

    # If --benchmark requested
    if args.benchmark:
        sample_f = Path(args.input) if args.input else (config.SAMPLE_DIR / "sample.fasta")
        ref_f = Path(args.reference)
        run_benchmark(sample_f, ref_f, num_benchmark_reads=100)
        if not args.input:
            return

    # STEP 3 & 4: Input validation / Interactive resolution
    if not args.input:
        args = interactive_prompt(args)

    input_path = Path(args.input)
    ref_path = Path(args.reference)

    if not input_path.exists():
        console.print(f"[bold red]Error: Input file does not exist: {input_path}[/bold red]")
        sys.exit(1)
    if not ref_path.exists():
        console.print(f"[bold red]Error: Reference file does not exist: {ref_path}[/bold red]")
        sys.exit(1)

    console.print(f"\n[bold green]▶ Configuring Pipeline Execution...[/bold green]")
    console.print(f"  • Input Reads File:  [cyan]{input_path}[/cyan]")
    console.print(f"  • Reference Genome:  [cyan]{ref_path}[/cyan]")
    console.print(f"  • Compute Target:    [cyan]{args.device.upper()}[/cyan]")
    console.print(f"  • Batch Chunk Size:  [cyan]{args.chunk_size:,}[/cyan]")
    console.print(f"  • Prefetch Workers:  [cyan]{args.workers}[/cyan]")

    # STEP 5 - 10: Run Alignment Pipeline
    t_start = time.time()
    pipeline = AlignmentPipeline(
        reference_path=ref_path,
        device_mode=args.device,
        chunk_size=args.chunk_size,
        num_workers=args.workers,
        match_score=config.DEFAULT_ALIGNMENT_CONFIG.match_score,
        mismatch_score=config.DEFAULT_ALIGNMENT_CONFIG.mismatch_score,
        gap_penalty=config.DEFAULT_ALIGNMENT_CONFIG.gap_penalty,
        enable_dashboard=not args.no_dashboard
    )

    results = pipeline.run(input_path)
    total_time = max(0.001, time.time() - t_start)

    # STEP 11 & 12: Reporting
    csv_file = config.OUTPUT_DIR / config.DEFAULT_REPORT_CONFIG.output_csv_name
    vcf_file = config.OUTPUT_DIR / config.DEFAULT_REPORT_CONFIG.output_vcf_name

    CSVReportGenerator.write_csv(results, csv_file)

    vcf_gen = VCFGenerator(
        reference_seq=pipeline.ref_seq,
        contig_id=pipeline.ref_id,
        min_depth=1,
        min_allele_freq=0.05
    )
    vcf_gen.write_vcf(results, vcf_file)

    # STEP 14: Final Summary Screen
    summary_table = Table(title="🎉 PIPELINE EXECUTION SUMMARY", box=None, header_style="bold green")
    summary_table.add_column("Property", style="bold cyan")
    summary_table.add_column("Result / Path", style="white")

    total_reads = len(results)
    aligned_reads = sum(1 for r in results if r.score > 0)
    avg_identity = sum(r.identity_pct for r in results) / max(1, total_reads)
    throughput = total_reads / total_time

    summary_table.add_row("Sequences Ingested", f"{total_reads:,}")
    summary_table.add_row("Successfully Aligned", f"{aligned_reads:,} ({aligned_reads/max(1, total_reads)*100:.1f}%)")
    summary_table.add_row("Mean Alignment Identity", f"{avg_identity:.2f}%")
    summary_table.add_row("Total Wall Time", f"{total_time:.2f} seconds")
    summary_table.add_row("Overall Throughput", f"{throughput:,.1f} reads/second")
    summary_table.add_row("CSV Alignment Report", f"{csv_file}")
    summary_table.add_row("VCF 4.2 Variant Report", f"{vcf_file}")

    console.print("\n")
    console.print(Panel(summary_table, border_style="green"))
    console.print("[bold green]✔ All alignment and variant calling operations completed successfully.[/bold green]\n")


if __name__ == "__main__":
    main()
