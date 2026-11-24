"""
Real-Time Terminal Dashboard Module
Renders dynamic live progress, hardware telemetry, throughput metrics, and ETA using Rich.
"""

import time
import os
import psutil
from typing import Optional
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live
from rich.text import Text


class TerminalDashboard:
    """
    Live terminal monitoring dashboard for the genomic alignment pipeline.
    Provides live throughput, memory consumption, GPU telemetry, and ETA updates.
    """

    def __init__(
        self,
        mode: str = "CPU",
        cuda_available: bool = False,
        device_name: str = "Host CPU",
        total_sequences: int = 0
    ):
        self.console = Console()
        self.mode = mode
        self.cuda_available = cuda_available
        self.device_name = device_name
        self.total_sequences = total_sequences
        
        self.processed_sequences = 0
        self.start_time = time.time()
        self.current_speed_rps = 0.0
        self.process = psutil.Process(os.getpid())

        # Progress bar component
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, style="cyan", complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn()
        )
        self.task_id = self.progress.add_task("Aligning Reads", total=self.total_sequences)
        self.live: Optional[Live] = None

    def start(self) -> "TerminalDashboard":
        self.start_time = time.time()
        self.live = Live(self._generate_display(), console=self.console, refresh_per_second=4)
        self.live.start()
        return self

    def stop(self) -> None:
        if self.live:
            self.live.stop()

    def update(self, batch_size: int) -> None:
        """Updates dashboard with newly completed batch."""
        self.processed_sequences += batch_size
        elapsed = max(0.001, time.time() - self.start_time)
        self.current_speed_rps = self.processed_sequences / elapsed
        
        self.progress.update(self.task_id, completed=self.processed_sequences)
        if self.live:
            self.live.update(self._generate_display())

    def _generate_display(self) -> Group:
        """Constructs layout panels and telemetry tables."""
        elapsed = time.time() - self.start_time
        remaining_seqs = max(0, self.total_sequences - self.processed_sequences)
        eta_seconds = (remaining_seqs / self.current_speed_rps) if self.current_speed_rps > 0 else 0

        # Memory telemetry
        mem_rss_mb = self.process.memory_info().rss / (1024 * 1024)

        # Telemetry Table
        table = Table(box=None, expand=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold white")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold white")

        cuda_status = "[bold green]Active[/bold green]" if self.cuda_available else "[yellow]Unavailable (Fallback)[/yellow]"
        active_engine = f"[bold green]CUDA Accelerated[/bold green]" if self.mode.upper() == "CUDA" else "[bold yellow]Multi-Core CPU Fallback[/bold yellow]"

        table.add_row("Execution Mode", active_engine, "Hardware Target", self.device_name[:32])
        table.add_row("CUDA Status", cuda_status, "Process RAM", f"{mem_rss_mb:.1f} MB")
        table.add_row("Total Reads", f"{self.total_sequences:,}", "Reads Processed", f"{self.processed_sequences:,}")
        table.add_row("Remaining Reads", f"{remaining_seqs:,}", "Throughput", f"{self.current_speed_rps:.1f} reads/s")
        table.add_row("Elapsed Time", f"{elapsed:.1f}s", "Estimated ETA", f"{eta_seconds:.1f}s")

        header_text = Text()
        header_text.append("🧬 HIGH-THROUGHPUT GENOMIC SEQUENCE ALIGNER\n", style="bold magenta")
        header_text.append("CUDA/Numba Accelerated Genomic Processing Pipeline", style="dim")

        return Group(
            Panel(header_text, style="blue"),
            Panel(self.progress, style="cyan"),
            Panel(table, title="[bold]Real-Time Telemetry & Hardware Status[/bold]", style="dim blue")
        )
