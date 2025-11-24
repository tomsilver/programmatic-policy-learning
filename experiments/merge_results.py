"""Module for merging results from multiple experiments.

This script processes and combines results from various experimental
runs into a single consolidated output for analysis.
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    """Merge results from multiple experimental runs.

    This function reads individual result files, processes them, and
    combines them into a single output file for further analysis.
    """

    # Hardcode "shifted"
    base_dir = Path("logs") / "shifted"
    print(f"Merging under: {base_dir.resolve()}")

    if not base_dir.exists():
        print("ERROR: logs/shifted does not exist.")
        return

    # Loop over env folders (TwoPileNim, Chase, etc.)
    for env_dir in base_dir.iterdir():
        if not env_dir.is_dir():
            continue

        print(f"\n→ Environment: {env_dir.name}")

        # Loop over spec folders (5_30_3, 10_40_3, ...)
        for spec_dir in env_dir.iterdir():
            if not spec_dir.is_dir():
                continue

            print(f"   → Spec: {spec_dir.name}")

            # Collect all *_result.csv inside this spec directory
            csv_files = list(spec_dir.glob("*_result.csv"))

            if not csv_files:
                print(f"      No result CSVs found in {spec_dir}")
                continue

            dfs = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df["source"] = csv_file.name  # optional
                    dfs.append(df)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"      Failed to read {csv_file}: {e}")

            if not dfs:
                print(f"      No readable CSVs in {spec_dir}")
                continue

            merged_df = pd.concat(dfs, ignore_index=True)

            final_path = spec_dir / "final_results.csv"
            merged_df.to_csv(final_path, index=False)

            print(f"      Saved merged results → {final_path}")


if __name__ == "__main__":
    main()
