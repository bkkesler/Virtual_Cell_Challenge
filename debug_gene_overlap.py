#!/usr/bin/env python3
"""
Debug script to check gene overlap between datasets
"""

import pandas as pd
import scanpy as sc
from pathlib import Path
import numpy as np

# Dataset configurations
DATASET_CONFIGS = {
    'VCC_Training_Subset': 'data/processed/normalized/VCC_Training_Subset_normalized.h5ad',
    '264667_hepg2': 'data/processed/normalized/264667_hepg2_normalized.h5ad',
    '264667_jurkat': 'standardized_datasets/264667_jurkat_standardized.h5ad',
    'K562_essential': 'standardized_datasets/K562_essential_standardized.h5ad',
    '274751_tfko': 'standardized_datasets/274751_tfko.sng.guides.full.ct_standardized.h5ad'
}

def check_gene_overlap():
    """Check gene overlap between datasets"""
    print("🔍 CHECKING GENE OVERLAP BETWEEN DATASETS")
    print("=" * 50)
    
    gene_sets = {}
    
    # Load gene names from each dataset
    for name, path in DATASET_CONFIGS.items():
        if Path(path).exists():
            try:
                print(f"\n📂 Loading {name}...")
                # Just load the var info, not the full data
                adata = sc.read_h5ad(path, backed='r')
                genes = list(adata.var.index)
                gene_sets[name] = set(genes)
                print(f"   ✅ {len(genes)} genes")
                
                # Show sample genes
                print(f"   📋 Sample genes: {genes[:10]}")
                
                adata.file.close()
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"❌ Not found: {path}")
    
    if len(gene_sets) < 2:
        print("❌ Need at least 2 datasets to check overlap")
        return
    
    # Check pairwise overlaps
    print(f"\n📊 PAIRWISE OVERLAP ANALYSIS:")
    print("-" * 40)
    
    dataset_names = list(gene_sets.keys())
    overlap_matrix = {}
    
    for i, d1 in enumerate(dataset_names):
        overlap_matrix[d1] = {}
        for j, d2 in enumerate(dataset_names):
            if i == j:
                overlap = len(gene_sets[d1])
                overlap_matrix[d1][d2] = overlap
                print(f"📊 {d1} = {d2}: {overlap} genes (self)")
            else:
                overlap = len(gene_sets[d1] & gene_sets[d2])
                union = len(gene_sets[d1] | gene_sets[d2])
                jaccard = overlap / union if union > 0 else 0
                overlap_matrix[d1][d2] = overlap
                print(f"📊 {d1} ∩ {d2}: {overlap} genes ({jaccard:.1%} Jaccard)")
    
    # Find best overlap
    max_overlap = 0
    best_pair = None
    
    for i, d1 in enumerate(dataset_names):
        for j, d2 in enumerate(dataset_names):
            if i < j:  # Avoid duplicates
                overlap = overlap_matrix[d1][d2]
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_pair = (d1, d2)
    
    print(f"\n🎯 BEST OVERLAP: {best_pair} with {max_overlap} common genes")
    
    # Check intersection of all datasets
    if len(gene_sets) > 1:
        all_intersection = set.intersection(*gene_sets.values())
        print(f"🔍 ALL DATASETS INTERSECTION: {len(all_intersection)} genes")
        
        if len(all_intersection) > 0:
            print(f"📋 Sample common genes: {sorted(list(all_intersection))[:20]}")
        else:
            print("❌ No genes common to all datasets!")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    if len(all_intersection) >= 5000:
        print("✅ Excellent! Use all datasets with full intersection")
    elif max_overlap >= 10000:
        print(f"✅ Good! Use best pair: {best_pair}")
        print(f"   This gives {max_overlap} common genes for training")
    elif max_overlap >= 5000:
        print(f"⚠️ Moderate overlap. Consider using: {best_pair}")
        print(f"   This gives {max_overlap} common genes")
    else:
        print("❌ Poor gene overlap between datasets!")
        print("💡 Check if gene standardization worked correctly")
        print("💡 Consider using datasets individually")
    
    # Check for gene format issues
    print(f"\n🔍 GENE FORMAT ANALYSIS:")
    print("-" * 25)
    
    for name, genes in gene_sets.items():
        gene_list = list(genes)
        ensembl_count = sum(1 for g in gene_list if str(g).startswith('ENSG'))
        symbol_count = sum(1 for g in gene_list if str(g).isalpha() and not str(g).startswith('ENSG'))
        
        print(f"📊 {name}:")
        print(f"   Ensembl IDs: {ensembl_count} ({ensembl_count/len(gene_list)*100:.1f}%)")
        print(f"   Gene symbols: {symbol_count} ({symbol_count/len(gene_list)*100:.1f}%)")
        print(f"   Sample: {gene_list[:5]}")

if __name__ == "__main__":
    check_gene_overlap()
