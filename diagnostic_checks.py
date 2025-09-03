#!/usr/bin/env python3
"""
Diagnostic checks for cross-dataset submission
"""

import pandas as pd
import anndata as ad
import numpy as np
from pathlib import Path
import sys


def check_submission_file():
    """Check the cross-dataset submission file for issues"""
    
    print("🔍 DIAGNOSTIC CHECKS FOR CROSS-DATASET SUBMISSION")
    print("=" * 60)
    
    # Check if files exist
    submission_path = Path("vcc_cross_dataset_submission/cross_dataset_full_submission.h5ad")
    gene_names_path = Path("vcc_cross_dataset_submission/gene_names.txt")
    
    print("📁 File existence check:")
    print(f"   Submission file: {'✅' if submission_path.exists() else '❌'} {submission_path}")
    print(f"   Gene names file: {'✅' if gene_names_path.exists() else '❌'} {gene_names_path}")
    
    if not submission_path.exists():
        print("❌ Submission file not found!")
        return False
    
    # Check file size
    file_size_gb = submission_path.stat().st_size / (1024**3)
    print(f"   📊 File size: {file_size_gb:.2f} GB")
    
    if file_size_gb > 10:
        print("⚠️ Very large file - this might cause memory issues")
    elif file_size_gb < 0.1:
        print("⚠️ Very small file - might be incomplete")
    
    # Try to load the file
    print(f"\n📂 Loading submission file...")
    try:
        adata = ad.read_h5ad(submission_path)
        print(f"✅ Successfully loaded submission")
        print(f"   📊 Shape: {adata.shape}")
        print(f"   🧬 Genes: {adata.n_vars:,}")
        print(f"   🎪 Cells: {adata.n_obs:,}")
    except Exception as e:
        print(f"❌ Error loading submission: {e}")
        return False
    
    # Check data structure
    print(f"\n🔍 Data structure checks:")
    
    # Check obs (cell metadata)
    print(f"   📋 obs columns: {list(adata.obs.columns)}")
    if 'target_gene' not in adata.obs.columns:
        print("❌ Missing 'target_gene' column in obs!")
        return False
    
    # Check var (gene metadata)  
    print(f"   🧬 var columns: {list(adata.var.columns)}")
    print(f"   🧬 Gene names sample: {list(adata.var.index[:5])}")
    
    # Check expression matrix
    print(f"\n📊 Expression matrix checks:")
    X = adata.X
    if hasattr(X, 'toarray'):
        print(f"   💾 Matrix type: Sparse ({X.format})")
        X_sample = X[:100, :100].toarray()  # Small sample for checking
    else:
        print(f"   💾 Matrix type: Dense")
        X_sample = X[:100, :100]
    
    print(f"   📈 Expression range: {X_sample.min():.3f} to {X_sample.max():.3f}")
    print(f"   📊 Mean expression: {X_sample.mean():.3f}")
    print(f"   📊 Std expression: {X_sample.std():.3f}")
    
    # Check for problematic values
    has_nan = np.isnan(X_sample).any()
    has_inf = np.isinf(X_sample).any()
    has_negative = (X_sample < 0).any()
    
    print(f"   🔍 Has NaN values: {'❌' if has_nan else '✅'}")
    print(f"   🔍 Has Inf values: {'❌' if has_inf else '✅'}")
    print(f"   🔍 Has negative values: {'⚠️' if has_negative else '✅'}")
    
    if has_nan or has_inf:
        print("❌ Found problematic values in expression matrix!")
    
    # Check perturbations
    print(f"\n🎯 Perturbation checks:")
    perturbations = adata.obs['target_gene'].value_counts()
    print(f"   🎪 Number of perturbations: {len(perturbations)}")
    print(f"   🔬 Has non-targeting: {'✅' if 'non-targeting' in perturbations.index else '❌'}")
    
    print(f"   📊 Cell counts per perturbation:")
    for pert, count in perturbations.head(10).items():
        print(f"      {pert}: {count:,} cells")
    
    if len(perturbations) < 50:
        print("⚠️ Expected ~51 perturbations, found fewer")
    
    # Check gene names file
    if gene_names_path.exists():
        print(f"\n📝 Gene names file checks:")
        with open(gene_names_path, 'r') as f:
            gene_names_content = f.read().strip()
        
        file_genes = gene_names_content.split('\n')
        adata_genes = list(adata.var.index)
        
        print(f"   📊 File gene count: {len(file_genes)}")
        print(f"   📊 AnnData gene count: {len(adata_genes)}")
        print(f"   🔍 Gene names match: {'✅' if len(file_genes) == len(adata_genes) else '❌'}")
        
        if len(file_genes) != len(adata_genes):
            print("❌ Gene count mismatch between file and AnnData!")
        
        # Check for header or empty lines
        if len(file_genes) > 0:
            first_gene = file_genes[0]
            if first_gene.lower() in ['gene', 'gene_name', 'gene_id']:
                print("⚠️ Gene names file might have a header")
            
            empty_lines = sum(1 for gene in file_genes if gene.strip() == '')
            if empty_lines > 0:
                print(f"⚠️ Found {empty_lines} empty lines in gene names file")
    
    # Memory estimation for cell-eval
    print(f"\n💾 Memory estimation:")
    estimated_memory_gb = (adata.n_obs * adata.n_vars * 4) / (1024**3)  # 4 bytes per float32
    print(f"   📊 Estimated memory: {estimated_memory_gb:.2f} GB")
    
    if estimated_memory_gb > 8:
        print("⚠️ High memory usage - cell-eval might struggle")
        print("💡 Consider reducing cell counts or making sparse")
    elif estimated_memory_gb > 4:
        print("⚠️ Moderate memory usage - might need optimization")
    else:
        print("✅ Memory usage looks reasonable")
    
    # Check for required VCC format
    print(f"\n📋 VCC format checks:")
    
    # Expected VCC gene count
    try:
        vcc_genes = pd.read_csv("data/raw/single_cell_rnaseq/vcc_data/gene_names.csv", header=None)[0].values
        expected_gene_count = len(vcc_genes)
        print(f"   🧬 Expected VCC genes: {expected_gene_count}")
        print(f"   🧬 Submission genes: {adata.n_vars}")
        print(f"   🔍 Gene count match: {'✅' if adata.n_vars == expected_gene_count else '❌'}")
        
        if adata.n_vars != expected_gene_count:
            print(f"❌ Gene count mismatch! Expected {expected_gene_count}, got {adata.n_vars}")
            
            # Check which genes are missing/extra
            submission_genes = set(adata.var.index)
            vcc_genes_set = set(vcc_genes)
            
            missing_genes = vcc_genes_set - submission_genes
            extra_genes = submission_genes - vcc_genes_set
            
            print(f"   📊 Missing genes: {len(missing_genes)}")
            print(f"   📊 Extra genes: {len(extra_genes)}")
            
            if len(missing_genes) > 0:
                print(f"   🔍 Sample missing: {list(missing_genes)[:5]}")
            if len(extra_genes) > 0:
                print(f"   🔍 Sample extra: {list(extra_genes)[:5]}")
        
    except Exception as e:
        print(f"⚠️ Could not load VCC gene names: {e}")
    
    # Check expected perturbations
    try:
        vcc_perts = pd.read_csv("data/raw/single_cell_rnaseq/vcc_data/pert_counts_Validation.csv")
        expected_perts = set(vcc_perts['target_gene'].values) | {'non-targeting'}
        submission_perts = set(adata.obs['target_gene'].values)
        
        print(f"   🎯 Expected perturbations: {len(expected_perts)}")
        print(f"   🎯 Submission perturbations: {len(submission_perts)}")
        
        missing_perts = expected_perts - submission_perts
        extra_perts = submission_perts - expected_perts
        
        if len(missing_perts) > 0:
            print(f"❌ Missing perturbations: {missing_perts}")
        if len(extra_perts) > 0:
            print(f"⚠️ Extra perturbations: {extra_perts}")
        
        if len(missing_perts) == 0 and len(extra_perts) == 0:
            print("✅ All expected perturbations present")
            
    except Exception as e:
        print(f"⚠️ Could not check perturbations: {e}")
    
    print(f"\n📋 SUMMARY:")
    
    # Overall assessment
    issues = []
    if has_nan or has_inf:
        issues.append("Invalid values in expression matrix")
    if adata.n_vars != len(vcc_genes):
        issues.append("Gene count mismatch")
    if 'non-targeting' not in adata.obs['target_gene'].values:
        issues.append("Missing non-targeting control")
    if estimated_memory_gb > 8:
        issues.append("Very high memory usage")
    
    if len(issues) == 0:
        print("✅ No major issues detected!")
        print("💡 Submission looks ready for cell-eval prep")
    else:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print("💡 Fix these issues before running cell-eval prep")
    
    return len(issues) == 0


def suggest_fixes():
    """Suggest fixes for common issues"""
    
    print(f"\n🔧 COMMON FIXES:")
    print("1. If gene count is wrong:")
    print("   - Re-run inference with corrected gene mapping")
    print("   - Check that VCC gene names file is correct")
    
    print("2. If memory usage is too high:")
    print("   - Reduce cells per perturbation (e.g., max 500 per pert)")
    print("   - Convert to sparse matrix")
    print("   - Use prep script with optimization")
    
    print("3. If there are NaN/Inf values:")
    print("   - Check model predictions for invalid outputs")
    print("   - Add bounds checking (e.g., clip to [0.01, 10])")
    
    print("4. If perturbations are missing:")
    print("   - Check perturbation name matching")
    print("   - Verify embedding availability")
    
    print("5. If gene names file is wrong:")
    print("   - Regenerate gene names file from AnnData.var.index")
    print("   - Ensure no header or empty lines")


def main():
    """Main diagnostic function"""
    
    success = check_submission_file()
    
    if not success:
        suggest_fixes()
        return 1
    
    print(f"\n🎉 Submission looks good!")
    print(f"💡 Try running: python prep_cross_dataset_submission.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
