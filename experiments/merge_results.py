"""Module for merging results from experiments."""

from pathlib import Path

import pandas as pd

paths = list(Path("outputs").rglob("result.csv"))

dfs = [pd.read_csv(p) for p in paths]
final = pd.concat(dfs, ignore_index=True)

final.to_csv("final_results.csv", index=False)
print("Saved final_results.csv with", len(final), "rows.")
