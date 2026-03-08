# exp0_2: Temporal clustering of seed events

For each query we take **top-k** seed events, merge intervals that are within **T** seconds of each other, then measure **hit rate** (overlap with ground truth) and **total time** (sum of merged interval lengths).

## What it does

1. **Load seeds** from `seed_events_AVA100.json` (tries `previous_run/` then `ava100_analysis/`).
2. **Per query:**  
   - Build intervals from `seed_events` (`duration` = [start_sec, end_sec]).  
   - **Top k:** Take the first k seeds by `borda_score` (desc) if present, else by start time.  
   - **Merge:** Intervals whose gap ≤ T are merged (e.g. [200,210] and [220,250] with T=10 → [200,250]).
3. **Measures:**  
   - **Hit rate:** Fraction of queries (with valid GT) where at least one merged interval overlaps the GT time.  
   - **Total time:** Sum over all queries of the total length of merged intervals (and average per query).

## Parameters

| Name   | Values              | Description                          |
|--------|---------------------|--------------------------------------|
| Top k  | 10, 30, 50, 80      | Number of seed events per query      |
| T (s)  | 10, 20, 30, 60, 120 | Max gap (seconds) to merge intervals |

Defined in `config.py` (`TOP_K`, `THRESHOLD_T`).

## Inputs

- **Seed events:** First existing path in `config.SEED_PATHS`  
  - `ECML-PKDD/previous_run/seed_events_AVA100.json`  
  - `ECML-PKDD/ava100_analysis/seed_events_AVA100.json`
- **Ground truth:** `project_root/datas/AVA100/` (citytour, ego, traffic, wildlife).  
  If missing, hit rate is skipped; total time is still reported.

## Output

- **Console:** Tables of hit rate, total time (sec), and avg time per query (sec) for each (k, T).
- **File:** `temporal_cluster_results.json` in this folder (same numbers + metadata).
- **Seed-only (before expansion):** `seed_per_question.json`, `seed_tables.txt`, `seed_hit_table.csv`, `seed_time_table.csv` — hit and time per question for clustered seeds at (EXPAND_K, EXPAND_T). **`seed_grid_tables.txt`** — same T \\ k grid as clustering (hit rate, total time, avg time/query) for comparison with expansion.
- **Expansion:** `expansion_per_question.json`, `expansion_tables.txt`, `expansion_hit_table.csv`, `expansion_time_table.csv` — hit and time per question for fb/evo (columns depend on `--expansion`).
- **Expansion grid** (with `--grid`): same T \\ k tables as clustering but for fb and evo expansion; `expansion_grid_results.json`.

## Expansion (optional)

After clustering, the pipeline can **expand** from cluster boundary events using:

- **forward_backward** (fb)
- **evt_obj_evt** (evo)

Use `--expansion` to run one or both:

- `--expansion both` (default): run fb and evo
- `--expansion fb`: forward_backward only
- `--expansion evo`: evt_obj_evt only

Use **`--fb-mode`** to control forward_backward expansion:

- `--fb-mode full` (default): keep all temporally expanded events (prev/next).
- `--fb-mode selective`: keep only expanded events where the event **sees the query object** (open-world object detection via `grounding.py`). Lowers time per question while aiming to preserve hit rate.

Use **`--grid`** to run expansion for **every (k, T)** (same grid as clustering). This prints hit rate, total time (sec), and avg time/query (sec) tables in the same T \\ k format as the clustering report, for forward_backward and for evt_obj_evt. Results are saved to `expansion_grid_results.json`. This is slow (one expansion per query per (k,T)).

## How to run

From **ECML-PKDD**:

```bash
python -m test_method.exp0_2.main [--expansion both|fb|evo]
```

From **this folder** (exp0_2):

```bash
python run.py [--expansion both|fb|evo]
```

## Files

| File              | Role                                      |
|-------------------|-------------------------------------------|
| `config.py`       | Paths and constants (TOP_K, THRESHOLD_T)  |
| `time_utils.py`   | Time parsing, time_ref → segments         |
| `gt.py`           | Load AVA100 ground truth                  |
| `clustering.py`   | Intervals, top-k, merge, hit check, total |
| `report.py`       | Print tables, write JSON                  |
| `main.py`         | Load data, run loop, call report          |
| `run.py`          | Launcher when running from exp0_2         |
