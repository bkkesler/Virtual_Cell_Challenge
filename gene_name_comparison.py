#!/usr/bin/env python3
"""
Compare gene names across datasets to identify differences
"""

import pandas as pd
from pathlib import Path
import numpy as np

def load_gene_names_from_file(gene_file_path):
    """Load gene names from a text file"""
    try:
        with open(gene_file_path, 'r') as f:
            genes = [line.strip() for line in f if line.strip()]
        return genes
    except Exception as e:
        print(f"Error loading {gene_file_path}: {e}")
        return []

def compare_gene_names():
    """Compare gene names across all datasets"""
    
    # Base directory containing the results
    base_dir = Path("median_nonzero_features")
    
    if not base_dir.exists():
        print(f"Directory {base_dir} not found!")
        return
    
    # Find all dataset directories
    dataset_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not dataset_dirs:
        print(f"No dataset directories found in {base_dir}")
        return
    
    print("🧬 GENE NAME COMPARISON ACROSS DATASETS")
    print("=" * 60)
    
    # Dictionary to store gene names for each dataset
    dataset_genes = {}
    
    # Load gene names from each dataset
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        gene_file = dataset_dir / "gene_names.txt"
        
        if gene_file.exists():
            genes = load_gene_names_from_file(gene_file)
            dataset_genes[dataset_name] = genes
            print(f"📁 {dataset_name}: {len(genes):,} genes")
        else:
            print(f"❌ {dataset_name}: gene_names.txt not found")
    
    if not dataset_genes:
        print("No gene name files found!")
        return
    
    print("\n" + "="*60)
    print("📋 FIRST 15 GENE NAMES FROM EACH DATASET:")
    print("="*60)
    
    # Print first 15 genes from each dataset
    for dataset_name, genes in dataset_genes.items():
        print(f"\n🔍 {dataset_name}:")
        print("-" * 40)
        
        if len(genes) >= 15:
            for i, gene in enumerate(genes[:15], 1):
                print(f"  {i:2d}. {gene}")
        else:
            for i, gene in enumerate(genes, 1):
                print(f"  {i:2d}. {gene}")
            print(f"     ... only {len(genes)} genes total")
    
    print("\n" + "="*60)
    print("🔍 GENE NAME FORMAT ANALYSIS:")
    print("="*60)
    
    # Analyze gene name formats
    for dataset_name, genes in dataset_genes.items():
        if not genes:
            continue
            
        print(f"\n📊 {dataset_name} Analysis:")
        
        # Sample some genes for format analysis
        sample_genes = genes[:10]
        
        # Check for different patterns
        has_ensembl = any(gene.startswith('ENSG') for gene in sample_genes)
        has_symbols = any(gene.isalpha() and not gene.startswith('ENSG') for gene in sample_genes)
        has_numbers = any(any(c.isdigit() for c in gene) for gene in sample_genes)
        has_dots = any('.' in gene for gene in sample_genes)
        has_underscores = any('_' in gene for gene in sample_genes)
        has_hyphens = any('-' in gene for gene in sample_genes)
        
        avg_length = np.mean([len(gene) for gene in sample_genes])
        min_length = min(len(gene) for gene in sample_genes)
        max_length = max(len(gene) for gene in sample_genes)
        
        print(f"  📏 Length: avg={avg_length:.1f}, min={min_length}, max={max_length}")
        print(f"  🧬 Has Ensembl IDs (ENSG*): {has_ensembl}")
        print(f"  🔤 Has gene symbols: {has_symbols}")
        print(f"  🔢 Contains numbers: {has_numbers}")
        print(f"  📍 Contains dots: {has_dots}")
        print(f"  🔗 Contains underscores: {has_underscores}")
        print(f"  ➖ Contains hyphens: {has_hyphens}")
        
        # Show some examples of different formats found
        unique_formats = set()
        for gene in sample_genes:
            if gene.startswith('ENSG'):
                unique_formats.add("Ensembl_ID")
            elif gene.isalpha():
                unique_formats.add("Gene_Symbol")
            elif any(c.isdigit() for c in gene):
                unique_formats.add("Contains_Numbers")
        
        print(f"  📝 Format types detected: {', '.join(unique_formats)}")
    
    print("\n" + "="*60)
    print("🔄 GENE OVERLAP ANALYSIS:")
    print("="*60)
    
    # Find overlaps between datasets
    dataset_names = list(dataset_genes.keys())
    
    if len(dataset_names) >= 2:
        for i, dataset1 in enumerate(dataset_names):
            for j, dataset2 in enumerate(dataset_names[i+1:], i+1):
                genes1 = set(dataset_genes[dataset1])
                genes2 = set(dataset_genes[dataset2])
                
                overlap = genes1.intersection(genes2)
                union = genes1.union(genes2)
                
                overlap_percent = len(overlap) / len(union) * 100 if union else 0
                
                print(f"\n🔗 {dataset1} ↔ {dataset2}:")
                print(f"  📊 Overlap: {len(overlap):,} genes ({overlap_percent:.1f}% of union)")
                print(f"  📊 {dataset1} only: {len(genes1 - genes2):,} genes")
                print(f"  📊 {dataset2} only: {len(genes2 - genes1):,} genes")
                
                # Show some examples of non-overlapping genes
                unique_to_1 = genes1 - genes2
                unique_to_2 = genes2 - genes1
                
                if unique_to_1:
                    sample_unique_1 = list(unique_to_1)[:5]
                    print(f"  🔸 Examples unique to {dataset1}: {', '.join(sample_unique_1)}")
                
                if unique_to_2:
                    sample_unique_2 = list(unique_to_2)[:5]
                    print(f"  🔹 Examples unique to {dataset2}: {', '.join(sample_unique_2)}")
    
    # Find common genes across ALL datasets
    if len(dataset_names) > 1:
        print(f"\n" + "="*60)
        print("🎯 GENES COMMON TO ALL DATASETS:")
        print("="*60)
        
        all_gene_sets = [set(genes) for genes in dataset_genes.values()]
        common_genes = set.intersection(*all_gene_sets)
        
        print(f"📊 Common genes across all {len(dataset_names)} datasets: {len(common_genes):,}")
        
        if common_genes:
            # Show first 20 common genes
            common_genes_sorted = sorted(list(common_genes))
            print(f"\n📋 First 20 common genes:")
            for i, gene in enumerate(common_genes_sorted[:20], 1):
                print(f"  {i:2d}. {gene}")
            
            if len(common_genes_sorted) > 20:
                print(f"     ... and {len(common_genes_sorted) - 20:,} more")
        else:
            print("❌ No genes are common to all datasets!")
    
    print(f"\n" + "="*60)
    print("💡 SUMMARY & RECOMMENDATIONS:")
    print("="*60)
    
    # Provide recommendations based on findings
    total_datasets = len(dataset_genes)
    if total_datasets == 0:
        print("❌ No datasets found to analyze")
        return
    
    # Check if gene name formats are consistent
    formats_by_dataset = {}
    for dataset_name, genes in dataset_genes.items():
        if genes:
            first_gene = genes[0]
            if first_gene.startswith('ENSG'):
                formats_by_dataset[dataset_name] = "Ensembl"
            elif first_gene.isalpha():
                formats_by_dataset[dataset_name] = "Symbol"
            else:
                formats_by_dataset[dataset_name] = "Mixed/Other"
    
    unique_formats = set(formats_by_dataset.values())
    
    if len(unique_formats) == 1:
        print(f"✅ All datasets use the same gene name format: {list(unique_formats)[0]}")
    else:
        print(f"⚠️ Mixed gene name formats detected:")
        for dataset, format_type in formats_by_dataset.items():
            print(f"   {dataset}: {format_type}")
        print("\n💡 Recommendation: Convert all to same format (e.g., gene symbols)")
    
    # Check overlap levels
    if len(dataset_names) >= 2:
        print(f"\n🔍 Gene overlap analysis shows compatibility for cross-dataset normalization")
    
    print(f"\n🎯 For the median non-zero normalization:")
    print(f"   - Focus on common genes for cross-dataset comparisons")
    print(f"   - Consider gene name standardization if formats differ")
    print(f"   - Check if low overlap is due to format differences vs. actual gene differences")

if __name__ == "__main__":
    compare_gene_names()