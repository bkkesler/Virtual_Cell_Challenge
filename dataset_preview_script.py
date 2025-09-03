#!/usr/bin/env python3
"""
Dataset Preview and Metadata Explorer

This script examines h5ad files and shows examples of their metadata structure,
helping to understand how perturbation information is stored in each dataset.
"""

import os
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
from pathlib import Path
from collections import Counter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
sc.settings.verbosity = 0

class DatasetPreview:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        
    def find_h5ad_files(self):
        """Recursively find all .h5ad files in the directory structure"""
        h5ad_files = []
        for file_path in self.root_dir.rglob("*.h5ad"):
            # Skip very large files for quick preview (>2GB)
            if file_path.stat().st_size < 2 * 1024 * 1024 * 1024:  # 2GB
                h5ad_files.append(file_path)
            else:
                print(f"Skipping large file: {file_path.name} ({file_path.stat().st_size / (1024**3):.1f}GB)")
        return sorted(h5ad_files)
    
    def preview_dataset(self, file_path, max_examples=5):
        """Preview a single dataset and show metadata examples"""
        try:
            print("=" * 100)
            print(f"DATASET: {file_path.name}")
            print(f"PATH: {file_path.relative_to(self.root_dir)}")
            print(f"SIZE: {file_path.stat().st_size / (1024**2):.1f} MB")
            print("=" * 100)
            
            # Load the data
            adata = sc.read_h5ad(file_path)
            
            print(f"Shape: {adata.n_obs} observations × {adata.n_vars} variables")
            print()
            
            # Show observation metadata columns
            print("OBSERVATION METADATA COLUMNS:")
            print("-" * 50)
            for i, col in enumerate(adata.obs.columns):
                dtype = adata.obs[col].dtype
                n_unique = adata.obs[col].nunique()
                print(f"{i+1:2d}. {col:<25} | dtype: {str(dtype):<10} | unique values: {n_unique}")
            print()
            
            # Show variable metadata columns if available
            if hasattr(adata, 'var') and len(adata.var.columns) > 0:
                print("VARIABLE METADATA COLUMNS:")
                print("-" * 50)
                for i, col in enumerate(adata.var.columns):
                    dtype = adata.var[col].dtype
                    n_unique = adata.var[col].nunique()
                    print(f"{i+1:2d}. {col:<25} | dtype: {str(dtype):<10} | unique values: {n_unique}")
                print()
            
            # Show uns (unstructured) keys if available
            if hasattr(adata, 'uns') and adata.uns:
                print("UNSTRUCTURED METADATA (uns) KEYS:")
                print("-" * 50)
                for key in adata.uns.keys():
                    value_type = type(adata.uns[key]).__name__
                    if isinstance(adata.uns[key], (list, np.ndarray)):
                        length = len(adata.uns[key])
                        print(f"  {key:<25} | type: {value_type:<10} | length: {length}")
                    else:
                        print(f"  {key:<25} | type: {value_type}")
                print()
            
            # Examine each observation column in detail
            print("DETAILED COLUMN ANALYSIS:")
            print("-" * 50)
            
            for col in adata.obs.columns:
                print(f"\nCOLUMN: {col}")
                print(f"  Data type: {adata.obs[col].dtype}")
                print(f"  Non-null values: {adata.obs[col].notna().sum()}/{len(adata.obs[col])}")
                
                if adata.obs[col].dtype == 'object' or adata.obs[col].dtype.name == 'category':
                    # Show value counts for categorical/string data
                    value_counts = adata.obs[col].value_counts()
                    print(f"  Unique values: {len(value_counts)}")
                    
                    if len(value_counts) <= 20:
                        print("  All values:")
                        for val, count in value_counts.head(20).items():
                            percentage = (count / len(adata.obs)) * 100
                            print(f"    '{val}': {count} ({percentage:.1f}%)")
                    else:
                        print(f"  Top {max_examples} values:")
                        for val, count in value_counts.head(max_examples).items():
                            percentage = (count / len(adata.obs)) * 100
                            print(f"    '{val}': {count} ({percentage:.1f}%)")
                        print(f"    ... and {len(value_counts) - max_examples} more")
                    
                    # Check if this looks like perturbation data
                    self.analyze_perturbation_potential(col, value_counts)
                    
                else:
                    # Show statistics for numerical data
                    print(f"  Min: {adata.obs[col].min()}")
                    print(f"  Max: {adata.obs[col].max()}")
                    print(f"  Mean: {adata.obs[col].mean():.2f}")
                    
                    # Show some example values
                    examples = adata.obs[col].dropna().head(max_examples).tolist()
                    print(f"  Example values: {examples}")
            
            # Show a few example rows of the complete observation metadata
            print(f"\nEXAMPLE OBSERVATIONS (first {max_examples} rows):")
            print("-" * 50)
            example_obs = adata.obs.head(max_examples)
            
            # Display as a formatted table
            for idx, (row_idx, row) in enumerate(example_obs.iterrows()):
                print(f"\nObservation {idx + 1} (index: {row_idx}):")
                for col, val in row.items():
                    print(f"  {col:<25}: {val}")
            
            return True
            
        except Exception as e:
            print(f"ERROR processing {file_path.name}: {str(e)}")
            return False
    
    def analyze_perturbation_potential(self, column_name, value_counts):
        """Analyze if a column likely contains perturbation information"""
        perturbation_indicators = [
            'perturbation', 'pert', 'gene_target', 'target_gene', 
            'guide_target', 'knockout', 'ko', 'gene', 'symbol',
            'perturbation_target', 'target', 'guide', 'treatment',
            'intervention', 'condition', 'cell_type'
        ]
        
        control_terms = ['control', 'ctrl', 'vehicle', 'dmso', 'untreated', 
                        'wild', 'wt', 'baseline', 'normal', 'mock', 'empty',
                        'scrambled', 'non-targeting', 'neg', 'positive']
        
        col_lower = column_name.lower()
        
        # Check if column name suggests perturbation data
        is_perturbation_column = any(indicator in col_lower for indicator in perturbation_indicators)
        
        # Analyze the values
        gene_like_count = 0
        control_like_count = 0
        
        for val in value_counts.index:
            if pd.notna(val):
                val_str = str(val).strip().lower()
                
                # Check for control terms
                if any(term in val_str for term in control_terms):
                    control_like_count += 1
                
                # Check for gene-like patterns (3-15 chars, mostly uppercase)
                val_orig = str(val).strip()
                if (3 <= len(val_orig) <= 15 and 
                    val_orig.replace('_', '').replace('-', '').isalnum() and 
                    sum(c.isupper() for c in val_orig) > len(val_orig) * 0.3):
                    gene_like_count += 1
        
        # Provide analysis
        print(f"  PERTURBATION ANALYSIS:")
        print(f"    Column name suggests perturbations: {is_perturbation_column}")
        print(f"    Gene-like values: {gene_like_count}/{len(value_counts)}")
        print(f"    Control-like values: {control_like_count}/{len(value_counts)}")
        
        if is_perturbation_column or gene_like_count > 1:
            print(f"    *** LIKELY PERTURBATION COLUMN ***")
        
        if gene_like_count > 0:
            print(f"    Potential gene names:")
            gene_candidates = []
            for val in value_counts.index:
                if pd.notna(val):
                    val_orig = str(val).strip()
                    if (3 <= len(val_orig) <= 15 and 
                        val_orig.replace('_', '').replace('-', '').isalnum() and 
                        sum(c.isupper() for c in val_orig) > len(val_orig) * 0.3):
                        gene_candidates.append(val_orig)
            
            for gene in gene_candidates[:10]:  # Show first 10
                print(f"      {gene}")
            if len(gene_candidates) > 10:
                print(f"      ... and {len(gene_candidates) - 10} more")
    
    def run_preview(self, max_files=None):
        """Run the preview analysis"""
        print("DATASET PREVIEW AND METADATA EXPLORER")
        print("=" * 100)
        
        # Find h5ad files
        h5ad_files = self.find_h5ad_files()
        
        if max_files:
            h5ad_files = h5ad_files[:max_files]
        
        print(f"\nFound {len(h5ad_files)} h5ad files to preview")
        print(f"Will analyze: {', '.join([f.name for f in h5ad_files])}")
        print()
        
        successful = 0
        for file_path in h5ad_files:
            if self.preview_dataset(file_path):
                successful += 1
            print("\n" + "="*100 + "\n")
        
        print(f"Successfully analyzed {successful}/{len(h5ad_files)} files")

def main():
    # Set the root directory path
    root_dir = "D:/Virtual_Cell3"  # Adjust this path as needed
    
    # Alternative: use current directory if script is run from project root
    # root_dir = "."
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} not found!")
        print("Please update the root_dir variable in the main() function")
        return
    
    # Create previewer and run analysis
    previewer = DatasetPreview(root_dir)
    
    # Preview all files (set max_files=3 to limit for testing)
    previewer.run_preview(max_files=5)  # Remove max_files to see all

if __name__ == "__main__":
    main()
