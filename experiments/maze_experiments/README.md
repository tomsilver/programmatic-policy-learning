# Maze Approach Comparison Experiments

This directory contains experiments to compare SearchApproach and ExpertMazeApproach (Oracle) on all mazes.

## Files

### Main Scripts
- `compare_maze_approaches.py`: Compare approaches on all mazes with fixed outer margin
- `compare_outer_margins.py`: Compare approaches across different outer margin values
- `compare_search_approaches.py`: Non-Hydra version of the comparison script

### Configuration Files
- `conf/compare_config.yaml`: Config for `compare_maze_approaches.py`
- `conf/outer_margin_config.yaml`: Config for `compare_outer_margins.py`

### Output Files
- `comparison_results.txt`: Results from `compare_maze_approaches.py`
- `outer_margin_results.txt`: Results from `compare_outer_margins.py`
- `plots/`: Directory containing all generated visualizations

## Usage

### 1. Compare Approaches (Fixed Outer Margin)

Run the comparison on all mazes with a fixed outer margin:

```bash
python experiments/maze_experiments/compare_maze_approaches.py
```

### 2. Compare Across Outer Margins

Run the comparison across different outer margin values (1, 5, 10, 20, 30, 40, 50, 80, 100):

```bash
python experiments/maze_experiments/compare_outer_margins.py
```

This will show how search metrics change as the outer world gets larger.

### Custom Configuration

Override settings via command line:

**For `compare_maze_approaches.py`:**
```bash
# Change seed
python experiments/maze_experiments/compare_maze_approaches.py seed=456

# Change max steps
python experiments/maze_experiments/compare_maze_approaches.py max_steps=2000

# Change outer margin
python experiments/maze_experiments/compare_maze_approaches.py outer_margin=5

# Change output file
python experiments/maze_experiments/compare_maze_approaches.py output_file=my_results.txt
```

**For `compare_outer_margins.py`:**
```bash
# Test different outer margins
python experiments/maze_experiments/compare_outer_margins.py outer_margins=[1,5,10,20]

# Change seed
python experiments/maze_experiments/compare_outer_margins.py seed=456

# Change output file
python experiments/maze_experiments/compare_outer_margins.py output_file=my_margin_results.txt
```

### Multiple Configurations (Hydra Multirun)

Run experiments with multiple seeds:

```bash
python experiments/maze_experiments/compare_maze_approaches.py -m seed=123,456,789
python experiments/maze_experiments/compare_outer_margins.py -m seed=123,456,789
```

## Output Format

### `compare_maze_approaches.py` Output

Generates:
1. **Text file** with individual and summary statistics
2. **Visualizations** (saved to `plots/`):
   - `expansions_comparison.png`: Bar chart comparing expansions per maze
   - `evaluations_comparison.png`: Bar chart comparing evaluations per maze
   - `combined_comparison.png`: Side-by-side view of both metrics

### `compare_outer_margins.py` Output

Generates:
1. **Text file** with summary table and detailed results
2. **Visualizations** (saved to `plots/`):
   - `outer_margin_expansions.png`: Line plot of expansions vs outer margin
   - `outer_margin_evaluations.png`: Line plot of evaluations vs outer margin
   - `outer_margin_combined.png`: Side-by-side view of both metrics

All outputs include:
- Goal Reached (True/False or percentage)
- Num Evals (number of state evaluations)
- Num Expansions (number of node expansions)

## Configuration Parameters

### Common Parameters (both scripts)
- `seed`: Random seed (default: 123)
- `max_steps`: Maximum steps per episode (default: 1000)
- `maze_dir`: Directory containing maze files (default: "data/mazes")
- `enable_render`: Enable visualization (default: false)
- `output_file`: Path to output results file

### `compare_maze_approaches.py` Specific
- `outer_margin`: Outer margin for maze environment (default: 10)

### `compare_outer_margins.py` Specific
- `outer_margins`: List of outer margins to test (default: [1, 5, 10, 20, 30, 40, 50, 80, 100])
