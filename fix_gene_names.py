#!/usr/bin/env python3
"""
Fix the gene names file for cross-dataset submission
"""

import anndata as ad
from pathlib import Path

def fix_gene_names():
    """Fix the gene names file to match AnnData"""
    
    print("🔧 FIXING GENE NAMES FILE")
    print("=" * 30)
    
    # Load the submission
    submission_path = Path("vcc_cross_dataset_submission/cross_dataset_full_submission.h5ad")
    gene_names_path = Path("vcc_cross_dataset_submission/gene_names.txt")
    
    print("📂 Loading submission...")
    adata = ad.read_h5ad(submission_path)
    
    print(f"📊 AnnData genes: {adata.n_vars}")
    print(f"🧬 Sample genes: {list(adata.var.index[:5])}")
    
    # Check for 'gene' header issue
    genes = list(adata.var.index)
    if genes[0] == 'gene':
        print("⚠️ Found 'gene' header - removing it")
        genes = genes[1:]  # Remove the header
        
        # Update the AnnData object
        adata = adata[:, 1:].copy()  # Remove first column
        print(f"✅ Fixed AnnData shape: {adata.shape}")
        
        # Save corrected AnnData
        corrected_path = submission_path.parent / "cross_dataset_full_submission_fixed.h5ad"
        adata.write_h5ad(corrected_path)
        print(f"✅ Saved corrected submission: {corrected_path}")
    
    # Write correct gene names file
    print("📝 Writing corrected gene names file...")
    with open(gene_names_path, 'w') as f:
        for gene in adata.var.index:
            f.write(f"{gene}\n")
    
    print(f"✅ Fixed gene names file: {len(adata.var.index)} genes")
    print(f"📁 File: {gene_names_path}")
    
    # Verify
    with open(gene_names_path, 'r') as f:
        file_genes = f.read().strip().split('\n')
    
    print(f"🔍 Verification:")
    print(f"   AnnData genes: {len(adata.var.index)}")
    print(f"   File genes: {len(file_genes)}")
    print(f"   Match: {'✅' if len(adata.var.index) == len(file_genes) else '❌'}")
    
    if genes[0] == 'gene':
        print(f"\n📤 Use the corrected file: cross_dataset_full_submission_fixed.h5ad")
        return str(corrected_path)
    else:
        print(f"\n📤 Original file is fine: cross_dataset_full_submission.h5ad")
        return str(submission_path)

if __name__ == "__main__":
    final_path = fix_gene_names()
    print(f"\n🎉 Ready for cell-eval prep!")
    print(f"📁 File to use: {final_path}")
