"""Data processing and I/O orchestration for trajectory graph analysis."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from graph_analysis.analyzer import TrajectoryGraphAnalyzer


class GraphMetricsExtractor:
    """Extracts and flattens metrics from graph data."""

    DIFFICULTY_MAPPING = {
        "<15 min fix": "under15min",
        "15 min - 1 hour": "under1h",
        "1-4 hours": "under4h",
        ">4 hours": "over4h"
    }

    @staticmethod
    def extract_metadata(graph_data: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata from graph structure."""
        graph_meta = graph_data.get("graph", {})

        debug_difficulty_raw = graph_meta.get("debug_difficulty", "unknown")
        debug_difficulty = GraphMetricsExtractor.DIFFICULTY_MAPPING.get(
            debug_difficulty_raw, debug_difficulty_raw
        )

        return {
            "instance": graph_meta.get("instance_name"),
            "resolution": graph_meta.get("resolution_status", "unknown"),
            "debug_difficulty": debug_difficulty,
        }

    @staticmethod
    def flatten_patch_stats(patch_stats: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested patch statistics."""
        flat_stats = patch_stats.copy()

        # Flatten fail_streaks
        fail_streaks = flat_stats.pop("fail_streaks", {})
        flat_stats["fail_streak_max"] = fail_streaks.get("max", 0)
        flat_stats["fail_streak_avg"] = fail_streaks.get("avg", 0)
        flat_stats["fail_streak_count"] = fail_streaks.get("count", 0)

        # Flatten fail_types
        fail_types = flat_stats.pop("fail_types", {})
        for fail_type, count in fail_types.items():
            flat_stats[f"fail_type_{fail_type}"] = count

        # Flatten fail_to_success_patterns
        patterns = flat_stats.get("fail_to_success_patterns", [])
        flat_stats["fail_to_success_patterns"] = str(patterns[0][0]) if patterns else "N/A"

        return flat_stats

    @staticmethod
    def analyze_graph_file(graph_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze a single graph file and return all metrics."""
        analyzer = TrajectoryGraphAnalyzer(graph_data)

        # Extract all metrics
        metrics = GraphMetricsExtractor.extract_metadata(graph_data)
        metrics.update(analyzer.get_metric_dict())
        metrics.update(analyzer.get_localization_summary())

        # Flatten and add patch statistics
        patch_stats = analyzer.get_patch_summary()
        flat_patch_stats = GraphMetricsExtractor.flatten_patch_stats(patch_stats)
        metrics.update(flat_patch_stats)

        return metrics


class TrajectoryAnalysisProcessor:
    """Orchestrates trajectory analysis workflow for a single graphs directory."""

    def __init__(self, graphs_dir: Path, output_dir: Path):
        self.graphs_dir = Path(graphs_dir)
        self.output_dir = Path(output_dir)

        if not self.graphs_dir.exists():
            raise FileNotFoundError(f"Graphs directory does not exist: {self.graphs_dir}")

    def process_all_graphs(self) -> pd.DataFrame:
        """Process all graph files in the graphs directory."""
        rows = []
        json_files = list(self.graphs_dir.rglob("*.json"))

        if not json_files:
            raise ValueError(f"No JSON files found in {self.graphs_dir}")

        for json_file in json_files:
            try:
                graph_data = self._load_graph_file(json_file)
                metrics = GraphMetricsExtractor.analyze_graph_file(graph_data)
                rows.append(metrics)
            except Exception as e:
                print(f"Warning: Failed to process {json_file}: {e}")
                continue

        if not rows:
            raise ValueError(f"No valid graph files processed from {self.graphs_dir}")

        return pd.DataFrame(rows)

    def _load_graph_file(self, file_path: Path) -> dict[str, Any]:
        """Load and parse a graph JSON file."""
        with open(file_path) as f:
            return json.load(f)

    def save_results(self, df: pd.DataFrame) -> Path:
        """Save analysis results to CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "trajectory_metrics.csv"
        df.to_csv(output_file, index=False)
        return output_file

    def run(self) -> Path:
        """Execute the complete analysis pipeline."""
        print(f"Processing graphs from: {self.graphs_dir}")
        df = self.process_all_graphs()
        print(f"Analyzed {len(df)} graph files")

        output_file = self.save_results(df)
        print(f"Results saved to: {output_file}")
        return output_file
