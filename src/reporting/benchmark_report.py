"""
Benchmark Visualization Module
Generates high-resolution performance comparison charts using Matplotlib.
"""

from pathlib import Path
from typing import List, Dict, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import pandas as pd


class BenchmarkVisualizer:
    """
    Renders comparative performance visualizations (execution time, throughput, speedup).
    """

    @staticmethod
    def generate_charts(benchmark_csv: str | Path, output_dir: str | Path) -> List[Path]:
        csv_path = Path(benchmark_csv)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not csv_path.exists():
            return []

        df = pd.read_csv(csv_path)
        if df.empty:
            return []

        generated_files: List[Path] = []

        # Style setup
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        colors = ["#2b5c8f", "#2ca02c", "#d62728", "#9467bd"]

        # 1. Execution Time Chart
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(df["Engine"], df["Execution_Time_s"], color=colors[:len(df)], width=0.5)
        ax.set_ylabel("Execution Time (seconds) - Lower is better", fontsize=11, fontweight="bold")
        ax.set_title("Genomic Sequence Alignment: Execution Time Comparison", fontsize=13, fontweight="bold", pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}s",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        chart_time = out_dir / "benchmark_execution_time.png"
        fig.savefig(chart_time, dpi=200)
        plt.close(fig)
        generated_files.append(chart_time)

        # 2. Throughput (Reads/sec) Chart
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(df["Engine"], df["Throughput_reads_sec"], color=colors[:len(df)], width=0.5)
        ax.set_ylabel("Throughput (Reads / Second) - Higher is better", fontsize=11, fontweight="bold")
        ax.set_title("Alignment Throughput Comparison", fontsize=13, fontweight="bold", pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:,.0f} r/s",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        chart_throughput = out_dir / "benchmark_throughput.png"
        fig.savefig(chart_throughput, dpi=200)
        plt.close(fig)
        generated_files.append(chart_throughput)

        # 3. Speedup Ratio Chart
        if "Speedup_Ratio" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(df["Engine"], df["Speedup_Ratio"], color=colors[:len(df)], width=0.5)
            ax.set_ylabel("Speedup Multiplier vs Pure Python Baseline", fontsize=11, fontweight="bold")
            ax.set_title("Acceleration Speedup Factor", fontsize=13, fontweight="bold", pad=12)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.1f}x",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontweight="bold")
            plt.tight_layout()
            chart_speedup = out_dir / "benchmark_speedup.png"
            fig.savefig(chart_speedup, dpi=200)
            plt.close(fig)
            generated_files.append(chart_speedup)

        return generated_files
