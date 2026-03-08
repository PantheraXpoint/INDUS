#!/usr/bin/env bash
# Exp0_1: Run seed truncation + expansion for top-K in {10,20,...,80}, then plot.
# Root: ECML-PKDD/test_method/exp0_1
# Usage: ./run_all.sh [ava100|lvbench|both]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="${1:-both}"

echo "=== Step 1: Truncate seeds and filter existing expanded files by top-K (no re-expansion) ==="
python run_seed_budget.py --dataset "$DATASET"

echo ""
echo "=== Step 2: Compute metrics and plot (binary overlap, percentage overlap, total time) ==="
python plot_seed_budget.py --dataset "$DATASET"

echo ""
if [ "$DATASET" = "both" ]; then
  echo "Done. Check plots in: $SCRIPT_DIR/plots/ava100/ and $SCRIPT_DIR/plots/lvbench/"
else
  echo "Done. Check plots in: $SCRIPT_DIR/plots/$DATASET/"
fi
