# Exp0_1: Seed budget vs hit rate and total time

**Goal:** Reduce the number of seeds to expand while maintaining hit rate. For each top-K (10, 20, …, 80), we truncate seeds by Borda score, run expansion (FB, EOE, EOOE), then compute binary overlap, percentage overlap, and total time.

**Root path:** `ECML-PKDD/test_method/exp0_1`

## Inputs (unchanged)

- **AVA100:** `ECML-PKDD/ava100_retrieval/` must contain:
  - `seed_events_AVA100.json` (full seeds)
  - `seed_events_AVA100_expanded_forward_backward_1.json`, `_evt_obj_evt_1.json`, `_evt_obj_obj_evt_1.json`
  - `seed_events_AVA100_expansion_trace_forward_backward_1.json`, `_evt_obj_evt_1.json`, `_evt_obj_obj_evt_1.json`
- **LVBench:** same layout under `ECML-PKDD/lvbench_retrieval/`.

## Scripts

1. **run_seed_budget.py**  
   - **Does not run expansion.** It reuses the existing expanded files and trace files in `{dataset}_retrieval/`.  
   - Truncates each query’s seeds to top-N by Borda score (N = 10, 20, …). Seeds with **no** `borda_score` are always included.  
   - For each top-K, writes truncated seeds to `exp0_1/{dataset}/top{N}/seed_events_{DATASET}.json`.  
   - For each expansion mode (FB, EOE, EOOE), **filters** the existing expanded file to only events reachable from the top-K seeds (using the expansion trace). Writes the filtered expanded file to `exp0_1/{dataset}/top{N}/`. So no call to `expand_indus_seed_events.py`.

2. **plot_seed_budget.py**  
   - For each (top_k, expansion) loads the expanded JSON, calls `calc_retrieval_accuracy.run_accuracy` for binary and percentage overlap, and computes total unique event duration (sum and mean per query).  
   - Plots: binary overlap vs seeds kept, percentage overlap vs seeds kept, total time (mean and total) vs seeds kept.  
   - Saves figures and a text report under `exp0_1/plots/{dataset}/`.

3. **run_all.sh**  
   - Runs `run_seed_budget.py` then `plot_seed_budget.py` for the given dataset.

## Usage

```bash
cd ECML-PKDD/test_method/exp0_1

# Both datasets, all top-K
./run_all.sh both

# One dataset
./run_all.sh ava100
./run_all.sh lvbench

# Or run steps separately
python run_seed_budget.py --dataset ava100
python plot_seed_budget.py --dataset ava100

# Subset of top-K
python run_seed_budget.py --dataset ava100 --top-k 10 20 40 80
python plot_seed_budget.py --dataset ava100 --top-k 10 20 40 80

# Skip re-running expansion if files exist
python run_seed_budget.py --dataset ava100 --skip-existing
```

## Output layout

```
exp0_1/
  ava100/
    top10/   seed_events_AVA100.json (truncated), seed_events_AVA100_expanded_*_1.json
    top20/   ...
    ...
  lvbench/
    top10/   ...
    ...
  plots/
    ava100/   exp0_1_binary_overlap_vs_seeds.png, exp0_1_percentage_overlap_vs_seeds.png,
              exp0_1_total_time_vs_seeds.png, exp0_1_efficiency_by_mode.png, exp0_1_report.txt
    lvbench/ ...
```

## Metrics (same as analyze_type)

- **Binary overlap accuracy:** fraction of queries with valid time GT that have at least one retrieved event overlapping GT.  
- **Percentage overlap accuracy:** mean over those queries of (fraction of GT time covered by retrieved events).  
- **Mean time per query:** mean unique event duration (sec) per query.  
- **Mean events per query:** mean number of unique retrieved events per query (for efficiency).

Report and plots include **base** (seeds only, no expansion) plus the three expansion modes (FB, EOE, EOOE).

**Efficiency by mode (`exp0_1_efficiency_by_mode.png`):** 2×4 subplots — one column per mode. Row 1: accuracy vs time/query; row 2: accuracy vs events/query. Points = top-K.

**Combined comparison (`exp0_1_efficiency_combined.png`):** One figure, 2×2 subplots, to compare all expansion approaches and both cost axes. Row 1: binary accuracy vs time (left) and vs events (right). Row 2: percentage accuracy vs time (left) and vs events (right). Each panel has 4 curves (base, FB, EOE, EOOE); same color/marker per mode across panels. Annotations = top-K. Use this to pick the best mode and top-K tradeoff at a glance.
