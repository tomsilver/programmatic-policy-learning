import os
import sys
from pathlib import Path
import pandas as pd

def main():
    ts = os.environ.get("EXPERIMENT_TS")
    if ts is None:
        print("ERROR: EXPERIMENT_TS not set.", file=sys.stderr)
        sys.exit(1)

    base_dir = Path("logs") / ts
    print(f"Merging results under experiment directory: {base_dir}")

    if not base_dir.exists():
        print(f"ERROR: Directory does not exist: {base_dir}", file=sys.stderr)
        sys.exit(1)

    # Iterate through each environment directory
    for env_dir in base_dir.iterdir():
        if not env_dir.is_dir():
            continue

        print(f"\nProcessing environment folder: {env_dir.name}")

        result_files = list(env_dir.rglob("result.csv"))

        if not result_files:
            print(f"No result.csv files found inside {env_dir}")
            continue

        rows = []
        for file in result_files:
            try:
                df = pd.read_csv(file)
                rows.append(df)
            except Exception as e:
                print(f"Failed to read {file}: {e}")

        if not rows:
            print(f"No readable result.csv files inside {env_dir}")
            continue

        # Merge results for this ENV only
        merged = pd.concat(rows, ignore_index=True)

        # Save per-environment merged CSV
        final_path = env_dir / "final_results.csv"
        merged.to_csv(final_path, index=False)

        print(f"Saved merged CSV for {env_dir.name} → {final_path}")

if __name__ == "__main__":
    main()

