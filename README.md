# INDUS: Context-Aware Graph Retrieval for Long-Video Question Answering

INDUS is a graph-based retrieval system for long-video question answering that improves evidence selection by combining:

- **query-aware retrieval**
- **content-aware verification**
- **cross-query context reuse**

The system is designed to retrieve a **minimal yet sufficient** set of supporting events from a video knowledge graph, reducing irrelevant context while improving downstream answer generation.

## Highlights

- Query-adaptive retrieval with shared Top-K seed selection
- Complementary **event-centric** and **object/entity-centric** verification
- **CrossCache** for cross-query reuse of verified temporal clusters
- Support for evaluation on **AVA100** and **LVBench**
- Scripts for retrieval evaluation, cache analysis, ablation, and visualization

## Method Overview

INDUS contains two main components:

### 1. QARVE
**Query-Adaptive Retrieval and Verification Engine** retrieves candidate graph nodes and filters them using query-aware verification.

It includes:
- **Event-centric traversal**, which expands temporally related events
- **Object/entity-centric traversal**, which verifies candidates using query-conditioned object cues
- **Context fusion**, which combines the verified outputs into the final retrieved context

### 2. CrossCache
**Cross-query Context Cache** stores verified temporal clusters for each video and reuses them across related queries.

Cache reuse is triggered when a seed cluster from the current query has sufficient temporal overlap with a cached cluster, measured by interval IoU.

## Repository Structure

A typical layout is:

```text
.
├── ECML-PKDD/
│   ├── calc_retrieval_accuracy.py
│   ├── expand_indus_seed_events.py
│   ├── merged_indus_boundary.py.py
│   ├── ava100_retrieval/
│   └── lvbench_retrieval/
├── datas/
│   ├── AVA100/
│   └── LVBench/
├── expansion_od/
├── embeddings/
├── llms/
└── figures/

## Run retrieval

AVA100:
```bash
python ECML-PKDD/merged_indus_boundary.py \
  --dataset AVA100 \
  --top-k-seeds 30 \
  --budget 80 \
  --merge-mode union \
  --fb-mode selective \
  --cache ECML-PKDD/cache/cache_ava100.json \
  --cache-overlap-threshold 0.5 \
  --cache-cluster-gap 30.0 \
  --final-cluster-gap 30.0
````

LVBench:

```bash
python ECML-PKDD/merged_indus_boundary.py \
  --dataset LVBench \
  --top-k-seeds 30 \
  --budget 80 \
  --merge-mode union \
  --fb-mode selective \
  --cache ECML-PKDD/cache/cache_lvbench.json \
  --cache-overlap-threshold 0.5 \
  --cache-cluster-gap 30.0 \
  --final-cluster-gap 30.0
```

## Evaluate retrieval

AVA100:

```bash
python ECML-PKDD/calc_retrieval_accuracy.py \
  --dataset AVA100 \
  --seed-events ECML-PKDD/ava100_retrieval/indus_explore/seed_events_AVA100_merged_final_union_selective_0.5_30.0_sharedcache_timeiou.json
```

LVBench:

```bash
python ECML-PKDD/calc_retrieval_accuracy.py \
  --dataset LVBench \
  --seed-events ECML-PKDD/lvbench_retrieval/indus_explore/seed_events_LVBench_merged_final_union_selective_0.5_30.0_sharedcache_timeiou.json
```

## Main settings

* Top-K seeds: `30`
* Cache overlap threshold: `0.5`
* Cache cluster gap: `30.0`
* GPU: single RTX 3090

