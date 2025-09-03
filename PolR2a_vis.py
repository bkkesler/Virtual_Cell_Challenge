#!/usr/bin/env python3
"""
Compute per-cell POLR2A count distributions across datasets.

Outputs per dataset:
- CSV: polr2a_counts.csv  (obs_name, polr2a_count)
- CSV: summary_stats.csv   (n_cells, detected_frac, mean, median, nonzero_mean, max)
- PNG: polr2a_hist.png     (histogram of counts; log1p on x if heavy-tailed)

Usage:
    python polr2a_distributions.py --base-dir "D:/Virtual_Cell3" --out "outputs/polr2a_distributions"
"""

import argparse
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

# --------- Helpers ---------
def find_h5ad_files(base_dir: Path) -> list[Path]:
    """Collect plausible single-cell AnnData files (prioritize standardized)."""
    candidates = []
    # Highest priority: standardized (symbols)
    std = base_dir / "standardized_datasets"
    if std.exists():
        candidates += sorted(std.glob("*.h5ad"))
    # Processed normalized (often symbols)
    proc_norm = base_dir / "data" / "processed" / "normalized"
    if proc_norm.exists():
        candidates += sorted(proc_norm.glob("*.h5ad"))
    # Raw single-cell (may be Ensembl IDs)
    raw_sc = base_dir / "data" / "raw" / "single_cell_rnaseq"
    if raw_sc.exists():
        candidates += sorted(raw_sc.glob("*.h5ad"))
    # Other submissions (OPTIONAL – often predictions, skip by default)
    # submissions = base_dir.glob("*submission*/**/*.h5ad")
    # candidates += [p for p in submissions if p.is_file()]
    # Deduplicate by filename
    seen = set()
    unique = []
    for p in candidates:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)
    return unique

def load_symbol_to_ensembl_map(base_dir: Path):
    """Return mapping dict: SYMBOL -> list of Ensembl IDs (and reverse) if available."""
    # Try common mapping artifacts in the project
    mapping_dir = base_dir / "gene_mappings"
    m = {}
    rev = {}
    try:
        # Prefer pickle if present
        for fname in ["ensembl_to_symbol_mapping.pkl", "NCBI_Gene_mapping.pkl", "HGNC_mapping.pkl"]:
            f = mapping_dir / fname
            if f.exists():
                mp = pd.read_pickle(f)
                # Try to normalize formats
                if isinstance(mp, dict):
                    ens_to_sym = mp
                    for ens, sym in ens_to_sym.items():
                        if isinstance(sym, (list, tuple, set)):
                            for s in sym:
                                if isinstance(s, str):
                                    m.setdefault(s.upper(), set()).add(ens)
                        elif isinstance(sym, str):
                            m.setdefault(sym.upper(), set()).add(ens)
                else:
                    # dataframe-like
                    df = pd.DataFrame(mp)
                    cols = [c.lower() for c in df.columns]
                    df.columns = cols
                    sym_col = next((c for c in df.columns if "symbol" in c), None)
                    ens_col = next((c for c in df.columns if "ensembl" in c), None)
                    if sym_col and ens_col:
                        for _, r in df[[sym_col, ens_col]].dropna().iterrows():
                            m.setdefault(str(r[sym_col]).upper(), set()).add(str(r[ens_col]))
        # Reverse
        for sym, enss in m.items():
            for ens in enss:
                rev.setdefault(ens, set()).add(sym)
    except Exception:
        pass
    # Convert sets to sorted lists
    m = {k: sorted(v) for k, v in m.items()}
    rev = {k: sorted(v) for k, v in rev.items()}
    return m, rev

def pick_matrix(adata: ad.AnnData):
    """Prefer a raw counts layer if present; else use X."""
    # Common layer names that store counts
    for lyr in ("counts", "raw", "umi", "raw_counts"):
        if lyr in adata.layers:
            return adata.layers[lyr], lyr
    # Some AnnData objects keep raw in .raw
    if adata.raw is not None:
        return adata.raw.X, "raw.X"
    return adata.X, "X"

def find_polr2a_index(adata: ad.AnnData, sym_to_ens: dict[str, list[str]] | None = None):
    """Find column index for POLR2A (symbol), with fallbacks for Ensembl IDs."""
    # 1) direct symbol match (case-insensitive)
    varnames_upper = pd.Index([str(v).upper() for v in adata.var_names])
    hits = np.where(varnames_upper == "POLR2A")[0]
    if hits.size > 0:
        return int(hits[0]), adata.var_names[hits[0]]

    # 2) look for a 'gene_symbols' column or similar
    for col in adata.var.columns:
        if "symbol" in col.lower():
            colvals = adata.var[col].astype(str).str.upper()
            hits = np.where(colvals == "POLR2A")[0]
            if hits.size > 0:
                return int(hits[0]), adata.var_names[hits[0]]

    # 3) If var_names are Ensembl-like, use mapping
    if sym_to_ens and all(name.upper().startswith(("ENSG", "ENST", "ENSMUSG", "ENSDARG")) for name in varnames_upper[:100]):
        enss = sym_to_ens.get("POLR2A", [])
        if enss:
            set_var = set(adata.var_names)
            for ens in enss:
                if ens in set_var:
                    idx = int(np.where(adata.var_names == ens)[0][0])
                    return idx, ens

    return None, None

def extract_column(matrix, col_idx: int) -> np.ndarray:
    """Return dense 1D array of a single column from X/layer (handles sparse)."""
    if sp.issparse(matrix):
        # matrix: (n_cells x n_genes)
        col = matrix[:, col_idx]
        return np.asarray(col.toarray()).ravel()
    # Dense ndarray or backed ndarray
    return np.asarray(matrix[:, col_idx]).ravel()

def compute_summary(vec: np.ndarray) -> dict:
    """Basic distribution summary."""
    n = vec.size
    detected = (vec > 0).sum()
    return {
        "n_cells": int(n),
        "detected_frac": float(detected / n if n else 0.0),
        "mean": float(vec.mean() if n else 0.0),
        "median": float(np.median(vec) if n else 0.0),
        "nonzero_mean": float(vec[vec > 0].mean() if detected > 0 else 0.0),
        "max": float(vec.max() if n else 0.0),
    }

def safe_name(path: Path) -> str:
    """Readable dataset name from file path."""
    return path.stem

# --------- Main workflow ---------
def main(base_dir: Path, out_dir: Path, log1p_hist: bool = True, bins: int = 60):
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = find_h5ad_files(base_dir)
    if not datasets:
        print("No .h5ad files found. Check your base directory.")
        return

    sym_to_ens, _ = load_symbol_to_ensembl_map(base_dir)
    index_file = []

    for ds in datasets:
        ds_name = safe_name(ds)
        ds_out = out_dir / ds_name
        ds_out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Processing: {ds_name} ===\n{ds}")

        try:
            # Use backed mode to avoid loading huge matrices into memory
            adata = sc.read_h5ad(ds, backed="r")
        except Exception as e:
            print(f"[WARN] Failed to open {ds}: {e}")
            continue

        try:
            idx, matched_name = find_polr2a_index(adata, sym_to_ens)
            if idx is None:
                print(f"[WARN] POLR2A not found in {ds_name}. Skipping.")
                continue

            Xlike, source = pick_matrix(adata)
            counts = extract_column(Xlike, idx)

            # Save per-cell counts
            obs_names = (adata.obs_names if adata.isbacked else pd.Index(range(counts.size)))
            df = pd.DataFrame({"obs": np.array(obs_names), "polr2a_count": counts})
            df.to_csv(ds_out / "polr2a_counts.csv", index=False)

            # Summary stats
            stats = compute_summary(counts)
            pd.DataFrame([stats]).to_csv(ds_out / "summary_stats.csv", index=False)
            with open(ds_out / "summary_stats.json", "w") as f:
                json.dump(stats, f, indent=2)

            # Histogram (optionally log1p x-axis)
            plt.figure(figsize=(6, 4))
            vals = np.log1p(counts) if log1p_hist else counts
            plt.hist(vals, bins=bins, density=True)  # ← density=True
            plt.xlabel("log1p(POLR2A counts)" if log1p_hist else "POLR2A counts")
            plt.ylabel("Density")  # ← updated label
            plt.title(f"{ds_name} — POLR2A per-cell distribution\n(source: {source})")
            plt.tight_layout()
            plt.savefig(ds_out / "polr2a_hist.png", dpi=150)
            plt.close()

            # Index record
            index_file.append({
                "dataset": ds_name,
                "path": str(ds),
                "var_match": matched_name,
                "matrix_source": source,
                **stats
            })
            print(f"[OK] Saved: {ds_out}")

        except Exception as e:
            print(f"[ERROR] {ds_name}: {e}")
        finally:
            try:
                # Close backed file
                adata.file.close()
            except Exception:
                pass

    # Global index
    if index_file:
        idx_df = pd.DataFrame(index_file).sort_values("dataset")
        idx_df.to_csv(out_dir / "index_summary.csv", index=False)
        print(f"\nWrote global summary: {out_dir / 'index_summary.csv'}")
    else:
        print("\nNo datasets produced results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True, help="Project root (e.g., D:/Virtual_Cell3)")
    parser.add_argument("--out", type=str, default="outputs/polr2a_distributions", help="Output directory")
    parser.add_argument("--no-log1p", action="store_true", help="Use raw counts on histogram x-axis")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins")
    args = parser.parse_args()

    main(
        base_dir=Path(args.base_dir),
        out_dir=Path(args.out),
        log1p_hist=not args.no_log1p,
        bins=args.bins,
    )
