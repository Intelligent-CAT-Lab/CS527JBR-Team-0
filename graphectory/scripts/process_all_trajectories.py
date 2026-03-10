#!/usr/bin/env python3
"""
Batch Process All Agent Trajectories

This script processes all SWE-agent and OpenHands trajectories for all models
and generates trajectory graphs in parallel.

Usage:
    python scripts/process_all_trajectories.py [--agents sa oh] [--workers 8] [--dry-run]

Examples:
    # Process all agents and models
    python scripts/process_all_trajectories.py

    # Process only SWE-agent trajectories
    python scripts/process_all_trajectories.py --agents sa

    # Process with 16 parallel workers
    python scripts/process_all_trajectories.py --workers 16

    # Dry run to see what would be processed
    python scripts/process_all_trajectories.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed


# ==================== Configuration ====================
MODEL_LIST = [
    "deepseek/deepseek-chat",
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/deepseek/deepseek-r1-0528",
    "openrouter/mistralai/devstral-small"
]

MODEL_TO_SHORT = {
    "deepseek/deepseek-chat": "dsk-v3",
    "openrouter/anthropic/claude-sonnet-4": "cld-4",
    "openrouter/deepseek/deepseek-r1-0528": "dsk-r1",
    "openrouter/mistralai/devstral-small": "dev"
}


@dataclass
class ProcessTask:
    """Represents a single graph generation task."""
    agent: str
    model: str
    model_short: str
    trajs_path: Path
    report_path: Path
    output_dir: Path

    def __str__(self):
        agent_name = "SWE-agent" if self.agent == "sa" else "OpenHands"
        return f"{agent_name}/{self.model_short}"


# ==================== Path Utilities ====================
def get_sa_model_name(model: str) -> str:
    """Convert model path to SWE-agent format: deepseek/deepseek-chat -> deepseek--deepseek-chat"""
    return model.replace("/", "--")


def get_oh_model_name(model: str) -> str:
    """Convert model path to OpenHands format: deepseek/deepseek-chat -> deepseek-chat"""
    return model.split("/")[-1]


def find_sa_trajectories(model: str, base_dir: Path) -> Tuple[Path, Path]:
    """Find SWE-agent trajectory and report paths.

    Args:
        model: Model identifier (e.g., "deepseek/deepseek-chat")
        base_dir: Base data directory

    Returns:
        Tuple of (trajectories_path, report_path)
    """
    sa_model_name = get_sa_model_name(model)
    oh_model_name = get_oh_model_name(model)

    # Trajectory path
    traj_dir_name = f"anthropic_filemap__{sa_model_name}__t-0.00__p-1.00__c-2.00___swe_bench_verified_test"
    trajs_path = base_dir / "SWE-agent" / "trajectories" / traj_dir_name

    # Report path
    report_path = base_dir / "SWE-agent" / "reports" / f"{oh_model_name}.json"

    return trajs_path, report_path


def find_oh_trajectories(model: str, base_dir: Path) -> Tuple[Path, Path]:
    """Find OpenHands trajectory and report paths.

    Args:
        model: Model identifier (e.g., "deepseek/deepseek-chat")
        base_dir: Base data directory

    Returns:
        Tuple of (trajectories_path, report_path)
    """
    oh_model_name = get_oh_model_name(model)

    # Trajectory directory
    traj_dir_name = f"{oh_model_name}_maxiter_100_N_v0.40.0-no-hint-run_1"
    traj_dir = base_dir / "OpenHands" / "trajectories" / traj_dir_name

    # output.jsonl path
    trajs_path = traj_dir / "output.jsonl"

    # Report path
    report_path = traj_dir / "report.json"

    return trajs_path, report_path


# ==================== Task Discovery ====================
def discover_tasks(agents: List[str], base_dir: Path) -> List[ProcessTask]:
    """Discover all processing tasks based on available data.

    Args:
        agents: List of agents to process ("sa", "oh")
        base_dir: Base data directory

    Returns:
        List of ProcessTask objects
    """
    tasks = []

    for model in MODEL_LIST:
        model_short = MODEL_TO_SHORT[model]

        if "sa" in agents:
            trajs_path, report_path = find_sa_trajectories(model, base_dir)
            if trajs_path.exists() and report_path.exists():
                tasks.append(ProcessTask(
                    agent="sa",
                    model=model,
                    model_short=model_short,
                    trajs_path=trajs_path,
                    report_path=report_path,
                    output_dir=base_dir
                ))
            else:
                print(f"[SKIP] SWE-agent/{model_short}: Missing data")
                if not trajs_path.exists():
                    print(f"       Missing: {trajs_path}")
                if not report_path.exists():
                    print(f"       Missing: {report_path}")

        if "oh" in agents:
            trajs_path, report_path = find_oh_trajectories(model, base_dir)
            if trajs_path.exists() and report_path.exists():
                tasks.append(ProcessTask(
                    agent="oh",
                    model=model,
                    model_short=model_short,
                    trajs_path=trajs_path,
                    report_path=report_path,
                    output_dir=base_dir
                ))
            else:
                print(f"[SKIP] OpenHands/{model_short}: Missing data")
                if not trajs_path.exists():
                    print(f"       Missing: {trajs_path}")
                if not report_path.exists():
                    print(f"       Missing: {report_path}")

    return tasks


# ==================== Task Execution ====================
def run_graph_generation(task: ProcessTask, script_path: Path) -> Dict:
    """Run graph generation for a single task.

    Args:
        task: ProcessTask to execute
        script_path: Path to generate_graphs.py script

    Returns:
        Result dictionary with status and output
    """
    cmd = [
        "python",
        str(script_path),
        "--agent", task.agent,
        "--model", task.model_short,
        "--trajs", str(task.trajs_path),
        "--eval_report", str(task.report_path),
        "--output_dir", str(task.output_dir)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per task
        )

        return {
            "task": str(task),
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        return {
            "task": str(task),
            "status": "timeout",
            "returncode": -1,
            "stdout": "",
            "stderr": "Process timed out after 1 hour"
        }
    except Exception as e:
        return {
            "task": str(task),
            "status": "error",
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)
        }


def process_tasks(tasks: List[ProcessTask], script_path: Path, max_workers: int) -> Dict:
    """Process all tasks in parallel.

    Args:
        tasks: List of ProcessTask objects
        script_path: Path to generate_graphs.py script
        max_workers: Maximum number of parallel workers

    Returns:
        Dictionary with success, failed, and timeout lists
    """
    results = {"success": [], "failed": [], "timeout": []}
    total = len(tasks)

    print(f"\n{'='*70}")
    print(f"Processing {total} tasks with {max_workers} workers...")
    print(f"{'='*70}\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_graph_generation, task, script_path): task
            for task in tasks
        }

        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_task):
            result = future.result()
            completed += 1

            status = result["status"]
            task_str = result["task"]

            if status == "success":
                results["success"].append(result)
                print(f"[{completed}/{total}] ✓ {task_str}")
            elif status == "timeout":
                results["timeout"].append(result)
                print(f"[{completed}/{total}] ⏱ {task_str} (timeout)")
            else:
                results["failed"].append(result)
                print(f"[{completed}/{total}] ✗ {task_str}")
                # Print error details for failed tasks
                if result["stderr"]:
                    print(f"          Error: {result['stderr'][:200]}")

    return results


# ==================== Main ====================
def print_summary(results: Dict):
    """Print final processing summary."""
    success = len(results["success"])
    failed = len(results["failed"])
    timeout = len(results["timeout"])
    total = success + failed + timeout

    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total:    {total}")
    print(f"Success:  {success}")
    print(f"Failed:   {failed}")
    print(f"Timeout:  {timeout}")
    print(f"{'='*70}\n")

    if failed > 0:
        print("Failed tasks:")
        for result in results["failed"]:
            print(f"  - {result['task']}")
            if result["stderr"]:
                print(f"    {result['stderr'][:150]}")
        print()

    if timeout > 0:
        print("Timed out tasks:")
        for result in results["timeout"]:
            print(f"  - {result['task']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all agent trajectories to generate graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all agents and models
  python scripts/process_all_trajectories.py

  # Process only SWE-agent
  python scripts/process_all_trajectories.py --agents sa

  # Process only OpenHands
  python scripts/process_all_trajectories.py --agents oh

  # Process with more workers
  python scripts/process_all_trajectories.py --workers 16

  # Dry run
  python scripts/process_all_trajectories.py --dry-run
        """
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["sa", "oh"],
        default=["sa", "oh"],
        help="Agents to process (default: both)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without executing"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: data)"
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    script_path = project_root / "graph_construction" / "generate_graphs.py"

    # Validate paths
    if not data_dir.exists():
        print(f"[ERROR] Data directory does not exist: {data_dir}")
        sys.exit(1)

    if not script_path.exists():
        print(f"[ERROR] Graph generation script not found: {script_path}")
        sys.exit(1)

    # Discover tasks
    print(f"Scanning for trajectories in: {data_dir}")
    tasks = discover_tasks(args.agents, data_dir)

    if not tasks:
        print("[ERROR] No tasks found. Check your data directory structure.")
        sys.exit(1)

    print(f"\nFound {len(tasks)} tasks to process:")
    for task in tasks:
        print(f"  - {task}")

    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN] Would execute the following commands:\n")
        for task in tasks:
            cmd = (
                f"python graph_construction/generate_graphs.py "
                f"--agent {task.agent} --model {task.model_short} "
                f"--trajs {task.trajs_path} "
                f"--eval_report {task.report_path} "
                f"--output_dir {task.output_dir}"
            )
            print(f"  {cmd}\n")
        sys.exit(0)

    # Process tasks
    results = process_tasks(tasks, script_path, args.workers)

    # Print summary
    print_summary(results)

    # Exit with appropriate code
    if results["failed"] or results["timeout"]:
        sys.exit(1)
    else:
        print("✓ All tasks completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
