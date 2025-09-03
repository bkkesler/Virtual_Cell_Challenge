#!/usr/bin/env python3
"""
Process raw VCC files to extract per-cell POLR2A counts and summaries.

Writes into:
  outputs/polr2a_distributions/VCC_Training_raw/
  outputs/polr2a_distributions/VCC_Training_subset_raw/
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

# ---- CONFIG ----
BASE = Path(r"D:\Virtual_Cell3")
RAW_VCC = BASE / r"data\raw\single_cell_rnaseq\vcc_data"
OUTDIR = BASE / r"outputs\polr2a_distributions"
FILES = {
    "VCC_Training_raw": RAW_VCC / "adata_Training.h5ad",
    "VCC_Training_subset_raw": RAW_VCC / "adata_Training_subset.h5ad",
}

PREFERRED_COUNT_LAYERS = ("counts", "umi", "raw_counts", "raw")

def pick_matrix(adata):
    for lyr in PREFERRED_COUNT_LAYERS:
        if lyr in adata.layers:
            return adata.layers[lyr], lyr
    if adata.raw is not None:
        return adata.raw.X, "raw.X"
    return adata.X, "X"

def find_polr2a_index(adata):
    # 1) direct symbol in var_names
    var_up = pd.Index([str(x).upper() for x in adata.var_names])
    hits = np.where(var_up == "POLR2A")[0]
    if hits.size:
        return int(hits[0]), adata.var_names[hits[0]]
    # 2) any symbol-like column
    for col in adata.var.columns:
        if "symbol" in col.lower() or col.lower() in ("gene", "genes", "gene_name", "gene_symbol"):
            colvals = adata.var[col].astype(str).str.upper()
            hits = np.where(colvals == "POLR2A")[0]
            if hits.size:
                return int(hits[0]), adata.var_names[hits[0]]
    # 3) alias/synonym columns
    for col in adata.var.columns:
        if any(k in col.lower() for k in ("hgnc", "alias", "synonym")):
            colvals = adata.var[col].astype(str).str.upper()
            hits = np.where(colvals == "POLR2A")[0]
            if hits.size:
                return int(hits[0]), adata.var_names[hits[0]]
    return None, None

def to_dense_1d(x):
    """Coerce any column-like into a dense 1-D float64 ndarray (handles sparse & matrices)."""
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x)             # handles HDF5 proxies, numpy.matrix, etc.
    x = np.ravel(x)               # 1-D
    x = x.astype(np.float64, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=None, neginf=None)
    return x

def extract_col(X, j):
    """Extract column j from X and return dense 1-D float array."""
    if sp.issparse(X):
        col = X[:, j]
    else:
        # backed arrays / h5py datasets also support slicing
        col = X[:, j]
    # If the result is sparse or matrix-like, force to dense 1-D
    if sp.issparse(col):
        return to_dense_1d(col)
    return to_dense_1d(col)

def summarize(vec):
    """Basic stats on dense 1-D vector."""
    vec = to_dense_1d(vec)
    n = vec.size
    nz = int(np.count_nonzero(vec))
    return {
        "n_cells": int(n),
        "detected_frac": float(nz / n) if n else 0.0,
        "mean": float(vec.mean()) if n else 0.0,
        "median": float(np.median(vec)) if n else 0.0,
        "nonzero_mean": float(vec[vec > 0].mean()) if nz else 0.0,
        "max": float(vec.max()) if n else 0.0,
    }

def process_one(label, path: Path):
    if not path.exists():
        print(f"[SKIP] {label}: file not found -> {path}")
        return False

    out = OUTDIR / label
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Processing RAW: {label} ===\n{path}")

    adata = sc.read_h5ad(path, backed="r")
    try:
        j, matched = find_polr2a_index(adata)
        if j is None:
            print(f"[WARN] {label}: POLR2A not found.")
            return False

        Xlike, src = pick_matrix(adata)
        counts = extract_col(Xlike, j)

        # Use a simple integer index to avoid backed obs_names length issues
        df = pd.DataFrame({
            "obs": np.arange(counts.size),
            "polr2a_count": counts
        })
        df.to_csv(out / "polr2a_counts.csv", index=False)

        stats = summarize(counts)
        pd.DataFrame([stats]).to_csv(out / "summary_stats.csv", index=False)
        (out / "summary_stats.json").write_text(json.dumps(stats, indent=2))

        # Histogram (log1p)
        plt.figure(figsize=(6, 4))
        plt.hist(np.log1p(counts), bins=60, density=True)  # ← density=True
        plt.xlabel("log1p(POLR2A counts)")
        plt.ylabel("Density")  # ← updated label
        plt.title(f"{label} — POLR2A distribution (source: {src})")
        plt.tight_layout()
        plt.savefig(out / "polr2a_hist.png", dpi=150)
        plt.close()

        # update/append to index_summary.csv
        idx_csv = OUTDIR / "index_summary.csv"
        row = {
            "dataset": label,
            "path": str(path),
            "var_match": matched,
            "matrix_source": src,
            **stats
        }
        if idx_csv.exists():
            idx = pd.read_csv(idx_csv)
            idx = pd.concat([idx, pd.DataFrame([row])], ignore_index=True)
            idx.drop_duplicates(subset=["dataset"], keep="last", inplace=True)
            idx.to_csv(idx_csv, index=False)
        else:
            pd.DataFrame([row]).to_csv(idx_csv, index=False)

        print(f"[OK] Saved: {out}")
        return True
    finally:
        try:
            adata.file.close()
        except Exception:
            pass

if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ok1 = process_one("VCC_Training_raw", FILES["VCC_Training_raw"])
    ok2 = process_one("VCC_Training_subset_raw", FILES["VCC_Training_subset_raw"])
    print("\nDone.")
