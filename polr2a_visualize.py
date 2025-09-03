#!/usr/bin/env python3
"""
Visualize per-cell POLR2A distributions produced by PolR2a_vis.py

Usage:
    python polr2a_visualize.py --in "outputs/polr2a_distributions" \
                               --max-cells 200000 \
                               --violin-max-per-dataset 50000
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ecdf(x: np.ndarray):
    x = np.sort(x)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

def load_counts_folder(folder: Path, max_cells: int | None = None) -> pd.DataFrame:
    csv = folder / "polr2a_counts.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv, usecols=["polr2a_count"])
    # optional subsample to keep plots snappy
    if max_cells is not None and len(df) > max_cells:
        df = df.sample(n=max_cells, random_state=42)
    df["dataset"] = folder.name
    df["log1p"] = np.log1p(df["polr2a_count"].astype(float).values)
    return df

def main(in_dir: Path, max_cells: int, violin_max_per_ds: int):
    idx_file = in_dir / "index_summary.csv"
    if not idx_file.exists():
        raise FileNotFoundError(f"index_summary.csv not found in {in_dir}")

    index_df = pd.read_csv(idx_file)
    # Keep only datasets that have a counts CSV present
    datasets = []
    for ds in index_df["dataset"].tolist():
        folder = in_dir / ds
        if (folder / "polr2a_counts.csv").exists():
            datasets.append(ds)

    if not datasets:
        raise SystemExit("No per-dataset polr2a_counts.csv files found.")

    # Load and concatenate (with per-dataset caps)
    frames = []
    for ds in datasets:
        df = load_counts_folder(in_dir / ds, max_cells=max_cells)
        if df.empty:
            continue
        # Additional per-dataset cap for violin (optional)
        if violin_max_per_ds and len(df) > violin_max_per_ds:
            df = df.sample(n=violin_max_per_ds, random_state=0)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    # Sort dataset order by detected fraction (nice for plots)
    det_map = dict(zip(index_df["dataset"], index_df.get("detected_frac", pd.Series([np.nan]*len(index_df)))))
    order = sorted(all_df["dataset"].unique(), key=lambda d: (-(det_map.get(d, np.nan)), d))

    # 1) Violin plot (log1p counts)
    plt.figure(figsize=(max(6, 0.6 * len(order)), 5))
    data_by_ds = [all_df.loc[all_df["dataset"] == d, "log1p"].values for d in order]
    plt.violinplot(data_by_ds, showextrema=True, showmeans=False)
    plt.xticks(range(1, len(order) + 1), order, rotation=45, ha="right")
    plt.ylabel("log1p(POLR2A counts)")
    plt.title("Per-dataset POLR2A distributions (log1p)")
    plt.tight_layout()
    out1 = in_dir / "combined_violin_log1p.png"
    plt.savefig(out1, dpi=150)
    plt.close()

    # 2) Detection fraction bar chart
    det_df = index_df[index_df["dataset"].isin(order)].copy()
    det_df = det_df.set_index("dataset").loc[order].reset_index()
    plt.figure(figsize=(max(6, 0.6 * len(order)), 4))
    plt.bar(det_df["dataset"], det_df["detected_frac"].astype(float).values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Detected fraction (POLR2A > 0)")
    plt.ylim(0, 1)
    plt.title("POLR2A detection fraction by dataset")
    plt.tight_layout()
    out2 = in_dir / "detection_fraction_bar.png"
    plt.savefig(out2, dpi=150)
    plt.close()

    # 3) Overlaid normalized histograms (log1p)
    plt.figure(figsize=(7, 5))
    for d in order:
        vals = all_df.loc[all_df["dataset"] == d, "log1p"].values
        plt.hist(vals, bins=60, histtype="step", density=True, label=d)
    plt.xlabel("log1p(POLR2A counts)")
    plt.ylabel("Density")
    plt.title("Overlaid POLR2A histograms (log1p)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out3 = in_dir / "overlaid_hist_log1p.png"
    plt.savefig(out3, dpi=150)
    plt.close()

    # 4) Overlaid ECDFs (log1p)
    plt.figure(figsize=(7, 5))
    for d in order:
        vals = all_df.loc[all_df["dataset"] == d, "log1p"].values
        x, y = ecdf(vals)
        plt.plot(x, y, label=d)
    plt.xlabel("log1p(POLR2A counts)")
    plt.ylabel("ECDF")
    plt.title("POLR2A ECDFs by dataset (log1p)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out4 = in_dir / "overlaid_ecdf_log1p.png"
    plt.savefig(out4, dpi=150)
    plt.close()

    print("Wrote:")
    for p in [out1, out2, out3, out4]:
        print(f" - {p}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=str, default="outputs/polr2a_distributions",
                    help="Folder containing index_summary.csv and per-dataset subfolders")
    ap.add_argument("--max-cells", type=int, default=200000,
                    help="Max cells to load per dataset (for speed/memory). Set 0 for no cap.")
    ap.add_argument("--violin-max-per-dataset", type=int, default=50000,
                    help="Additional per-dataset cap specifically for violin plot.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    max_cells = None if args.max_cells in (None, 0) else int(args.max_cells)
    main(in_dir, max_cells, int(args.violin_max_per_dataset))
