
#!/usr/bin/env python3
"""
GSE303901 Downloader & Organizer

- Downloads filtered feature-barcode matrices and guide metadata from GEO GSE303901
- Organizes into a clean project structure
- Optionally extracts .tar.gz archives
- Creates a MANIFEST.csv with file locations and sizes
- Includes a small utility to load extracted 10x matrices into AnnData

Usage examples:
    python download_gse303901.py --base-dir "D:/PerturbSeq_DC_TAP"
    python download_gse303901.py --extract --subset k562 --base-dir "./data"

Notes:
- URLs are from the GEO Series page (accessed Aug 17, 2025).
- If files already exist with correct size, they will be skipped.
"""

import argparse
import csv
import gzip
import hashlib
import io
import os
import sys
import tarfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict

import requests

# Optional: only required if you use the AnnData loader
try:
    import scanpy as sc
except Exception:
    sc = None


GEO_BASE = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE303901"
# File inventory taken from the GEO supplementary file table (as of Aug 17, 2025)
FILES = [
    # Pilot / benchmarking
    ("GSE303901_H-DC-TAP_filtered_feature_bc_matrix.tar.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_H-DC-TAP_filtered_feature_bc_matrix.tar.gz", "matrices", "pilot"),
    ("GSE303901_H-SG-TAP_filtered_feature_bc_matrix.tar.gz",  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_H-SG-TAP_filtered_feature_bc_matrix.tar.gz",  "matrices", "pilot"),
    ("GSE303901_L-SG-TAP_filtered_feature_bc_matrix.tar.gz",  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_L-SG-TAP_filtered_feature_bc_matrix.tar.gz",  "matrices", "pilot"),
    ("GSE303901_P16-scRNAseq_filtered_feature_bc_matrix.tar.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_P16-scRNAseq_filtered_feature_bc_matrix.tar.gz", "matrices", "pilot"),

    # Guide metadata
    ("GSE303901_P10-TAP_all_guide_info.txt.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_P10-TAP_all_guide_info.txt.gz", "metadata", "guides"),

    # K562 MOI series
    ("GSE303901_k562_1-MOI_filtered_feature_bc_matrix.tar.gz",  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_k562_1-MOI_filtered_feature_bc_matrix.tar.gz",  "matrices", "k562"),
    ("GSE303901_k562_3-MOI_filtered_feature_bc_matrix.tar.gz",  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_k562_3-MOI_filtered_feature_bc_matrix.tar.gz",  "matrices", "k562"),
    ("GSE303901_k562_6-MOI_filtered_feature_bc_matrix.tar.gz",  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_k562_6-MOI_filtered_feature_bc_matrix.tar.gz",  "matrices", "k562"),
    ("GSE303901_k562_9-MOI_filtered_feature_bc_matrix.tar.gz",  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_k562_9-MOI_filtered_feature_bc_matrix.tar.gz",  "matrices", "k562"),
    ("GSE303901_k562_14-MOI_filtered_feature_bc_matrix.tar.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_k562_14-MOI_filtered_feature_bc_matrix.tar.gz", "matrices", "k562"),

    # WTC11 MOI series
    ("GSE303901_wtc11_1_MOI_filtered_feature_bc_matrix.tar.gz",  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_wtc11_1_MOI_filtered_feature_bc_matrix.tar.gz",  "matrices", "wtc11"),
    ("GSE303901_wtc11_2_MOI_filtered_feature_bc_matrix.tar.gz",  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_wtc11_2_MOI_filtered_feature_bc_matrix.tar.gz",  "matrices", "wtc11"),
    ("GSE303901_wtc11_3_MOI_filtered_feature_bc_matrix.tar.gz",  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_wtc11_3_MOI_filtered_feature_bc_matrix.tar.gz",  "matrices", "wtc11"),
    ("GSE303901_wtc11_6_MOI_filtered_feature_bc_matrix.tar.gz",  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_wtc11_6_MOI_filtered_feature_bc_matrix.tar.gz",  "matrices", "wtc11"),
    ("GSE303901_wtc11_10_MOI_filtered_feature_bc_matrix.tar.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303901/suppl/GSE303901_wtc11_10_MOI_filtered_feature_bc_matrix.tar.gz", "matrices", "wtc11"),

    # Raw bundle (optional)
    ("GSE303901_RAW.tar", "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE303901&format=file&file=GSE303901_RAW.tar", "raw", "raw"),
]


@dataclass
class FileRec:
    filename: str
    url: str
    category: str     # 'matrices' | 'metadata' | 'raw'
    group: str        # 'pilot' | 'k562' | 'wtc11' | 'guides' | 'raw'


def sizeof_fmt(num, suffix="B"):
    for unit in ["","K","M","G","T","P","E","Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"


def http_head_size(url: str) -> Optional[int]:
    try:
        r = requests.head(url, timeout=20, allow_redirects=True)
        if r.ok:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl is not None and cl.isdigit() else None
    except Exception:
        return None
    return None


def stream_download(url: str, out_path: Path, expected_size: Optional[int] = None, chunk: int = 1<<20) -> None:
    """Download with streaming; skip if file exists and matches expected_size."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and expected_size is not None and out_path.stat().st_size == expected_size:
        print(f"[SKIP] {out_path.name} exists ({sizeof_fmt(out_path.stat().st_size)})")
        return

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        if expected_size is None and total > 0:
            expected_size = total
        bytes_done = 0
        t0 = time.time()
        with open(out_path, "wb") as f:
            for chunk_bytes in r.iter_content(chunk_size=chunk):
                if chunk_bytes:
                    f.write(chunk_bytes)
                    bytes_done += len(chunk_bytes)
                    if time.time() - t0 > 1.0:
                        t0 = time.time()
                        if expected_size:
                            pct = 100.0 * bytes_done / expected_size
                            sys.stdout.write(f"\r[DL] {out_path.name}: {sizeof_fmt(bytes_done)}/{sizeof_fmt(expected_size)} ({pct:.1f}%)")
                        else:
                            sys.stdout.write(f"\r[DL] {out_path.name}: {sizeof_fmt(bytes_done)}")
                        sys.stdout.flush()
        if expected_size and out_path.stat().st_size != expected_size:
            raise IOError(f"Size mismatch after download for {out_path} (got {out_path.stat().st_size} expected {expected_size})")
        print(f"\r[OK ] {out_path.name} ({sizeof_fmt(out_path.stat().st_size)})")


def extract_tar_gz(tar_gz_path: Path, dest_dir: Path) -> Optional[Path]:
    """Extract .tar.gz into dest_dir/<stem_without_extensions>"""
    try:
        # Determine extraction subdir
        stem = tar_gz_path.name
        for ext in [".tar.gz", ".tgz", ".tar"]:
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        outdir = dest_dir / stem
        outdir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_gz_path, "r:*") as tar:
            tar.extractall(outdir)
        print(f"[EXT] {tar_gz_path.name} -> {outdir}")
        return outdir
    except Exception as e:
        print(f"[WARN] Failed to extract {tar_gz_path.name}: {e}")
        return None


def write_manifest(records: List[Dict], manifest_path: Path) -> None:
    if not records:
        return
    keys = sorted(records[0].keys())
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"[OK ] Wrote manifest: {manifest_path}")


def list_inventory(subset: Optional[str]) -> List[FileRec]:
    """subset: None | 'k562' | 'wtc11' | 'pilot' | 'minimal'"""
    inv = []
    for fn, url, cat, grp in FILES:
        if subset is None or grp == subset or (subset == "minimal" and (grp in ("k562", "wtc11") or grp == "guides")):
            inv.append(FileRec(fn, url, cat, grp))
    return inv


def build_paths(base_dir: Path) -> Dict[str, Path]:
    layout = {
        "root": base_dir,
        "raw": base_dir / "data" / "raw" / "gse303901",
        "matrices": base_dir / "data" / "raw" / "gse303901" / "matrices",
        "metadata": base_dir / "data" / "raw" / "gse303901" / "metadata",
        "raw_bundle": base_dir / "data" / "raw" / "gse303901" / "raw",
        "processed": base_dir / "data" / "processed" / "gse303901",
        "manifests": base_dir / "data" / "manifests",
    }
    for p in layout.values():
        p.mkdir(parents=True, exist_ok=True)
    return layout


def load_10x_to_anndata(extracted_dir: Path, output_h5ad: Path) -> None:
    """
    Minimal example loader: if Scanpy is available and extracted_dir contains a 10x filtered_feature_bc_matrix,
    create an AnnData and save .h5ad
    """
    if sc is None:
        print("[INFO] scanpy not installed; skipping AnnData creation.")
        return

    # Look for a directory that has matrix.mtx(.gz), features.tsv(.gz), barcodes.tsv(.gz)
    def find_10x_root(p: Path) -> Optional[Path]:
        candidates = []
        for root, dirs, files in os.walk(p):
            files_set = set(files)
            if any(f.startswith("matrix.mtx") for f in files_set) and \
               any(f.startswith("features.tsv") for f in files_set) and \
               any(f.startswith("barcodes.tsv") for f in files_set):
                return Path(root)
        return None

    tenx_root = find_10x_root(extracted_dir)
    if tenx_root is None:
        print(f"[WARN] No 10x files found under {extracted_dir}")
        return

    print(f"[LOAD] 10x at {tenx_root}")
    adata = sc.read_10x_mtx(tenx_root.as_posix(), var_names="gene_symbols", cache=False)
    adata.write_h5ad(output_h5ad.as_posix())
    print(f"[OK ] Saved {output_h5ad}")


def main():
    parser = argparse.ArgumentParser(description="Download and organize GEO GSE303901 for Perturb-seq modeling.")
    parser.add_argument("--base-dir", type=str, default="./perturbseq_dc_tap", help="Base project directory")
    parser.add_argument("--subset", type=str, default=None, choices=[None, "k562", "wtc11", "pilot", "minimal"], help="Subset of files to download")
    parser.add_argument("--extract", action="store_true", help="Extract tar archives into processed/")
    parser.add_argument("--make-h5ad", action="store_true", help="Create .h5ad for each extracted 10x matrix (requires scanpy)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    paths = build_paths(base_dir)

    inventory = list_inventory(args.subset)
    manifest_rows = []

    for rec in inventory:
        target_dir = paths[rec.category if rec.category in paths else "raw"]
        if rec.category == "raw":
            target_dir = paths["raw_bundle"]
        out_path = target_dir / rec.filename
        expected_size = http_head_size(rec.url)
        try:
            stream_download(rec.url, out_path, expected_size=expected_size)
        except Exception as e:
            print(f"[ERR] Failed to download {rec.filename}: {e}")
            continue

        manifest_rows.append({
            "filename": rec.filename,
            "url": rec.url,
            "category": rec.category,
            "group": rec.group,
            "bytes": out_path.stat().st_size,
            "path": str(out_path),
        })

        if args.extract and rec.filename.endswith((".tar.gz", ".tgz", ".tar")) and rec.category != "metadata":
            extracted = extract_tar_gz(out_path, paths["processed"])
            if extracted and args.make_h5ad:
                # name .h5ad after the extracted dir
                h5ad = paths["processed"] / f"{extracted.name}.h5ad"
                load_10x_to_anndata(extracted, h5ad)

    write_manifest(manifest_rows, paths["manifests"] / "GSE303901_MANIFEST.csv")

    print("\nDone.")
    print(f"GEO page: {GEO_BASE}")
    print(f"Base directory: {base_dir}")
    print("Structure:")
    for k, p in paths.items():
        print(f"  - {k}: {p}")


if __name__ == "__main__":
    main()
