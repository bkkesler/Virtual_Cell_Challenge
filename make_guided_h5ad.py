
#!/usr/bin/env python3
"""
make_guided_h5ad.py

Create a guided AnnData (.h5ad) from a GEO GSE303901 tarball and the guide metadata file.

Usage (PowerShell / cmd):
  python make_guided_h5ad.py ^
    --tar "D:/Virtual_Cell3/perturbseq_dc_tap/data/raw/gse303901/matrices/GSE303901_k562_1-MOI_filtered_feature_bc_matrix.tar.gz" ^
    --guide "D:/Virtual_Cell3/perturbseq_dc_tap/data/raw/gse303901/metadata/GSE303901_P10-TAP_all_guide_info.txt.gz" ^
    --out "D:/Virtual_Cell3/perturbseq_dc_tap/data/processed/gse303901/k562_moi1_guided.h5ad" ^
    --onehot

Requires:
  pip install scanpy anndata pandas numpy
"""

import argparse
import os
import tarfile
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import scanpy as sc
except Exception as e:
    raise SystemExit("scanpy is required. Please 'pip install scanpy anndata'") from e


def extract_tar(tar_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(out_dir)
    # If the tar contains a single top-level folder, use it; else use out_dir
    entries = [p for p in out_dir.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return out_dir


def find_10x_root(p: Path) -> Path:
    for root, dirs, files in os.walk(p):
        s = set(files)
        if any(f.startswith("matrix.mtx") for f in s) and \
           any(f.startswith("features.tsv") for f in s) and \
           any(f.startswith("barcodes.tsv") for f in s):
            return Path(root)
    raise FileNotFoundError(f"No 10x files found under {p}")


def read_guides(guide_path: Path) -> pd.DataFrame:
    df = pd.read_csv(guide_path, sep=None, engine="python", low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    # Find barcode column
    bc_col = next((cols[k] for k in ["barcode","cell_barcode","cb","cell","cellid","cell_id","cell-barcode"] if k in cols), None)
    if bc_col is None:
        raise ValueError(f"Could not find a barcode column in {guide_path}. Columns: {list(df.columns)}")
    # Guide column
    guide_col = next((cols[k] for k in ["guide","guides","sgrna","grna","sg_rna"] if k in cols), None)
    # Target column
    target_col = next((cols[k] for k in ["target","gene","target_gene","targetgene","gene_symbol","symbol"] if k in cols), None)

    out = pd.DataFrame({"barcode": df[bc_col].astype(str).str.replace("-1$","",regex=True)})
    # Guides
    if guide_col is not None:
        out["guides"] = df[guide_col].astype(str)
    else:
        guide_like = [c for c in df.columns if c.lower().startswith("guide")]
        if guide_like:
            out["guides"] = df[guide_like].astype(str).apply(lambda r: ",".join([x for x in r if x and x.lower() != "nan"]), axis=1)
        else:
            out["guides"] = ""
    # Targets
    if target_col is not None:
        out["targets"] = df[target_col].astype(str)
    else:
        tar_like = [c for c in df.columns if any(c.lower().startswith(p) for p in ["target_gene","target","gene"])]
        if tar_like:
            out["targets"] = df[tar_like].astype(str).apply(lambda r: ",".join([x for x in r if x and x.lower() != "nan"]), axis=1)
        else:
            out["targets"] = ""

    # Normalize separators
    for col in ["guides","targets"]:
        out[col] = out[col].str.replace(r"[;|/]", ",", regex=True).str.replace(r"\s+", "", regex=True).str.strip(",")

    return out.drop_duplicates(subset=["barcode"], keep="last")


def onehot_multi(series: pd.Series, prefix: str) -> pd.DataFrame:
    tokens = series.fillna("").str.split(",")
    vocab = sorted({tok for toks in tokens for tok in toks if tok})
    if not vocab:
        return pd.DataFrame(index=series.index)
    arr = np.zeros((len(series), len(vocab)), dtype=np.int8)
    col_index = {v:i for i,v in enumerate(vocab)}
    for i, toks in enumerate(tokens):
        for t in toks:
            if t in col_index:
                arr[i, col_index[t]] = 1
    return pd.DataFrame(arr, index=series.index, columns=[f"{prefix}:{v}" for v in vocab])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, help="Path to *_filtered_feature_bc_matrix.tar.gz")
    ap.add_argument("--guide", required=True, help="Path to GSE303901_P10-TAP_all_guide_info.txt.gz")
    ap.add_argument("--out", required=True, help="Path to output .h5ad")
    ap.add_argument("--extract-dir", default=None, help="Optional directory for extraction (defaults next to --out)")
    ap.add_argument("--onehot", action="store_true", help="Add one-hot columns for guides/targets")
    args = ap.parse_args()

    tar_path = Path(args.tar).resolve()
    guide_path = Path(args.guide).resolve()
    out_path = Path(args.out).resolve()

    extract_dir = Path(args.extract_dir).resolve() if args.extract_dir else out_path.parent / (tar_path.stem.replace(".tar",""))
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXTRACT] {tar_path} -> {extract_dir}")
    extracted_root = extract_tar(tar_path, extract_dir)
    tenx_root = find_10x_root(extracted_root)
    print(f"[10X     ] {tenx_root}")

    print("[READ   ] loading 10x into AnnData ...")
    adata = sc.read_10x_mtx(tenx_root.as_posix(), var_names="gene_symbols", cache=False)

    print("[GUIDES ] reading guide metadata ...")
    gdf = read_guides(guide_path)

    # Merge
    adata.obs["barcode"] = [b.replace("-1","") for b in adata.obs_names.astype(str)]
    merged = pd.merge(pd.DataFrame({"barcode": adata.obs["barcode"]}), gdf, on="barcode", how="left")
    adata.obs["guides"]  = merged["guides"].fillna("")
    adata.obs["targets"] = merged["targets"].fillna("")

    if args.onehot:
        print("[ONEHOT ] building one-hot encodings ...")
        gh = onehot_multi(adata.obs["guides"], "guide")
        th = onehot_multi(adata.obs["targets"], "target")
        gh.index = adata.obs_names
        th.index = adata.obs_names
        for col in gh.columns:
            adata.obs[col] = gh[col].astype(np.int8)
        for col in th.columns:
            adata.obs[col] = th[col].astype(np.int8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path.as_posix())
    print(f"[OK     ] wrote {out_path}")

if __name__ == "__main__":
    main()
