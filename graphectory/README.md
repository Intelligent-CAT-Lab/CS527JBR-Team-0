# Graphectory

**Process-centric analysis of agentic software systems.**

Graphectory transforms agent execution traces into structured graphs that capture the problem-solving patterns of AI software engineering agents. By modeling agent actions as directed graphs with phase classification (localization, patching, validation), this tool enables systematic analysis of how agents approach and solve software engineering tasks.

---

<!-- ## Dataset

**Pre-computed Graphs**: Full dataset (2 agents × 4 models) available under [data/{OpenHands|SWE-agent}/graphs](data/)

**Raw Trajectories**: Hosted on Zenodo due to file size: [https://zenodo.org/records/17364210](https://zenodo.org/records/17364210)

--- -->

## Installation

```bash

cd graphectory
python -m pip install -e .
```

We recommend using conda or virtual environments (python>=3.8) to manage dependencies.

This [link](https://stackoverflow.com/questions/69970147/how-do-i-resolve-the-pygraphviz-error-on-mac-os) may be helpful if you encounter errors when installing pygraphviz on macOS.
If you are using Windows, we also recommend working in WSL.

---

## Quick Start

### Basic Usage

```bash
python graph_construction/generate_graphs.py \
  --agent {sa|oh} \
  --model {dsk-v3|dsk-r1|dev|cld-4|gpt-5-mini} \
  --trajs <path_to_trajectories> \
  --eval_report <path_to_report.json> \
  --output_dir <output_directory>
```

### Sample runs

**SWE-agent with GPT-5-mini:**
Below is the command to construct graphs for the 500 [SWE-agent (backboned with GPT-5-mini) trajectories](https://drive.google.com/file/d/1GCgz9klGjkipfwdkNxXmGznC9pnbBchs/view?usp=sharing)
```bash
python graph_construction/generate_graphs.py \
  --agent sa --model gpt-5-mini \
  --trajs ./Trajectories/gpt-5-mini-traj \
  --eval_report data/SWE-agent/reports/gpt-5-mini.json \
  --output_dir data/samples
```

**Output**: `{output_dir}/{Agent}/graphs/{model}/{instance_id}/{instance_id}.{json,pdf}`

---

## Input Requirements

| Argument | Description | Format |
|----------|-------------|--------|
| `--agent` | Agent type | `sa` (SWE-agent), `oh` (OpenHands) |
| `--model` | Model identifier | `dsk-v3`, `dsk-r1`, `dev`, `cld-4`, `gpt-5-mini` (extensible) |
| `--trajs` | Trajectory path | **SWE-agent**: directory with `.traj` files<br>**OpenHands**: `output.jsonl` file |
| `--eval_report` | Evaluation report | JSON with `resolved_ids`/`unresolved_ids` keys |
| `--output_dir` | Base output directory | Organized as `{agent}/graphs/{model}/{instance_id}/` |
| `--workers` | Parallel workers (optional) | Default: 8 |

---

## Graph Construction Process

1. **Parsing**: Agent trajectories → atomic actions (tool calls, commands, subcommands)
2. **Node Deduplication**: Identical actions merged with occurrence tracking
3. **Phase Classification**: Actions categorized using heuristics ([mapPhase.py](graph_construction/mapPhase.py)):
   - **Localization**: Information gathering, searching, test generation before patching
   - **Patch**: Creating/editing non-test files
   - **Validation**: Running tests or editing test files after patching
   - **General**: Other actions (planning, environment setup)
4. **Edge Construction**: Execution edges (sequential flow) + hierarchical edges (conceptual dependencies)
5. **Output**: JSON (NetworkX node-link format) + PDF visualization with phase-colored nodes

**Graph Metadata**: Each graph includes `resolution_status`, `instance_name`, and `debug_difficulty`

---

## Extending Graphectory

### Adding New Models

You can easily add new models without code modification:

```bash
python graph_construction/generate_graphs.py \
  --agent sa --model your-custom-model \
  --trajs <your_trajectories> \
  --eval_report <your_report> \
  --output_dir <output>
```

Simply provide a unique model identifier. Outputs will be organized under `graphs/your-custom-model/`.

### Supporting New SWE-agent Tools

To parse custom SWE-agent tools:

1. Add the tool's `config.yaml` to [graph_construction/generate_graphs.py:236-242](graph_construction/generate_graphs.py#L236-L242)
2. Update `tool_configs` list in `setup_parser_for_agent()`

Example:
```python
tool_configs = [
    "data/SWE-agent/tools/edit_anthropic/config.yaml",
    "data/SWE-agent/tools/your_custom_tool/config.yaml",  # Add here
]
```

### Supporting New Agents

To add support for a new agent framework:

1. **Implement trajectory loader** in [buildGraph.py](graph_construction/buildGraph.py) following the pattern:
   ```python
   def build_graph_from_newagent_trajectory(traj_data, parser, instance_id, output_dir, eval_report_path):
       builder = GraphBuilder()
       # Parse agent-specific trajectory structure
       # Convert to builder.add_or_update_node() calls
       return builder.finalize_and_save(output_dir, instance_id, eval_report_path)
   ```

2. **Add agent mapping** in [generate_graphs.py](graph_construction/generate_graphs.py):
   - Update `SUPPORTED_AGENTS` and `AGENT_NAMES`
   - Add conditional branch in `GraphProcessor.process_trajectory()`

**Key principle**: Different agents have different trajectory formats, but all generate the same unified graph structure (nodes with phases, execution/hierarchical edges, metadata).

---

## Graph Analysis

Pre-computed analysis results for the full dataset are available under [data/{OpenHands|SWE-agent}/analysis](data/), including Graphectory metrics.

### Analyze Pre-computed Graphs

```bash
python -m graph_analysis.batch_runner
```
<!-- 
### Analyze Custom Graphs

```bash
python -m graph_analysis.batch_runner --data-dir ./my_data --output-dir ./my_output
``` -->

Results are saved to `trajectory_metrics.csv`.

---