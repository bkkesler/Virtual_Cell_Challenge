# VCC Submission Diagnostic Tools
import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import h5py
from pathlib import Path
import subprocess

print("🔍 VCC SUBMISSION DIAGNOSTIC TOOLS")
print("=" * 40)

# Define paths
submission_dir = "./vcc_esm2_go_rf_v2_submission"
h5ad_file = "esm2_go_rf_v2_submission.h5ad"
vcc_file = "esm2_go_rf_v2_submission.prep.vcc"
gene_names_file = "gene_names.txt"

print(f"📁 Analyzing submission directory: {submission_dir}")

if not os.path.exists(submission_dir):
    print(f"❌ Submission directory not found: {submission_dir}")
    sys.exit(1)

os.chdir(submission_dir)

# =============================================================================
# 1. ANALYZE ORIGINAL H5AD FILE
# =============================================================================
print(f"\n📊 ANALYZING ORIGINAL H5AD FILE")
print("-" * 35)

if os.path.exists(h5ad_file):
    try:
        adata = ad.read_h5ad(h5ad_file)
        print(f"✅ Successfully loaded {h5ad_file}")
        print(f"   Shape: {adata.shape}")
        print(f"   Observations (cells): {adata.n_obs:,}")
        print(f"   Variables (genes): {adata.n_vars:,}")
        
        # Check obs columns
        print(f"   Obs columns: {list(adata.obs.columns)}")
        
        # Check perturbation column
        if 'target_gene' in adata.obs.columns:
            pert_counts = adata.obs['target_gene'].value_counts()
            print(f"   Perturbations: {len(pert_counts)}")
            print(f"   Top 5 perturbations:")
            for pert, count in pert_counts.head().items():
                print(f"     {pert}: {count:,} cells")
            
            # Check for non-targeting
            if 'non-targeting' in pert_counts:
                print(f"   ✅ Non-targeting found: {pert_counts['non-targeting']:,} cells")
            else:
                print(f"   ❌ Non-targeting not found!")
                print(f"   Available perturbations: {list(pert_counts.index[:10])}")
        else:
            print(f"   ❌ 'target_gene' column not found!")
            print(f"   Available columns: {list(adata.obs.columns)}")
        
        # Check var columns
        print(f"   Var columns: {list(adata.var.columns)}")
        print(f"   Gene names sample: {list(adata.var.index[:5])}")
        
        # Check data type and sparsity
        if hasattr(adata.X, 'toarray'):
            print(f"   Data type: Sparse ({type(adata.X)})")
            sample_data = adata.X[:1000].toarray()
            sparsity = np.mean(sample_data == 0)
            print(f"   Sparsity (sample): {sparsity:.2%}")
        else:
            print(f"   Data type: Dense ({type(adata.X)})")
            sample_data = adata.X[:1000]
            sparsity = np.mean(sample_data == 0)
            print(f"   Sparsity (sample): {sparsity:.2%}")
        
        # Check for negative values
        if hasattr(adata.X, 'toarray'):
            sample_min = adata.X[:1000].toarray().min()
        else:
            sample_min = adata.X[:1000].min()
        
        if sample_min < 0:
            print(f"   ⚠️ Contains negative values: min = {sample_min}")
        else:
            print(f"   ✅ All values non-negative: min = {sample_min}")
        
        # Check for NaN/inf
        if hasattr(adata.X, 'toarray'):
            sample_data = adata.X[:1000].toarray()
        else:
            sample_data = adata.X[:1000]
        
        has_nan = np.isnan(sample_data).any()
        has_inf = np.isinf(sample_data).any()
        
        if has_nan:
            print(f"   ⚠️ Contains NaN values")
        if has_inf:
            print(f"   ⚠️ Contains infinite values")
        
        if not has_nan and not has_inf:
            print(f"   ✅ No NaN or infinite values detected")
            
    except Exception as e:
        print(f"❌ Error loading {h5ad_file}: {e}")
        adata = None
else:
    print(f"❌ File not found: {h5ad_file}")
    adata = None

# =============================================================================
# 2. ANALYZE GENE NAMES FILE
# =============================================================================
print(f"\n📋 ANALYZING GENE NAMES FILE")
print("-" * 30)

if os.path.exists(gene_names_file):
    try:
        with open(gene_names_file, 'r') as f:
            gene_names = [line.strip() for line in f if line.strip()]
        
        print(f"✅ Gene names file loaded")
        print(f"   Total genes: {len(gene_names):,}")
        print(f"   First 5 genes: {gene_names[:5]}")
        print(f"   Last 5 genes: {gene_names[-5:]}")
        
        # Check for duplicates
        unique_genes = set(gene_names)
        if len(unique_genes) != len(gene_names):
            duplicates = len(gene_names) - len(unique_genes)
            print(f"   ⚠️ Duplicate genes found: {duplicates}")
        else:
            print(f"   ✅ All genes unique")
        
        # Check gene name format
        empty_names = sum(1 for name in gene_names if not name or name.isspace())
        if empty_names > 0:
            print(f"   ⚠️ Empty gene names: {empty_names}")
        
        # Check against h5ad file
        if adata is not None:
            h5ad_genes = list(adata.var.index)
            
            if len(gene_names) == len(h5ad_genes):
                print(f"   ✅ Gene count matches h5ad file")
                
                # Check if gene names match
                genes_match = gene_names == h5ad_genes
                if genes_match:
                    print(f"   ✅ Gene names perfectly match h5ad file")
                else:
                    mismatches = sum(1 for i in range(len(gene_names)) if gene_names[i] != h5ad_genes[i])
                    print(f"   ⚠️ Gene name mismatches: {mismatches}")
                    
                    # Show first few mismatches
                    print(f"   First few mismatches:")
                    for i in range(min(5, len(gene_names))):
                        if gene_names[i] != h5ad_genes[i]:
                            print(f"     Position {i}: '{gene_names[i]}' vs '{h5ad_genes[i]}'")
            else:
                print(f"   ❌ Gene count mismatch!")
                print(f"     Gene names file: {len(gene_names):,}")
                print(f"     H5AD file: {len(h5ad_genes):,}")
                
    except Exception as e:
        print(f"❌ Error reading gene names: {e}")
else:
    print(f"❌ Gene names file not found: {gene_names_file}")

# =============================================================================
# 3. ANALYZE VCC FILE (Basic info only)
# =============================================================================
print(f"\n📦 ANALYZING VCC FILE")
print("-" * 25)

if os.path.exists(vcc_file):
    file_size = os.path.getsize(vcc_file) / 1e6
    print(f"✅ VCC file exists: {vcc_file}")
    print(f"   Size: {file_size:.1f} MB")
    
    # Try to get basic info without fully reading
    try:
        # VCC files are typically HDF5 format
        with h5py.File(vcc_file, 'r') as f:
            print(f"   Format: HDF5")
            print(f"   Top-level keys: {list(f.keys())}")
            
            # Try to find common structures
            for key in f.keys():
                item = f[key]
                if hasattr(item, 'shape'):
                    print(f"     {key}: shape {item.shape}, dtype {item.dtype}")
                elif hasattr(item, 'keys'):
                    print(f"     {key}: group with {len(item.keys())} items")
                else:
                    print(f"     {key}: {type(item)}")
    
    except Exception as e:
        print(f"   ⚠️ Could not analyze VCC structure: {e}")
        print(f"   File may be corrupted or in unexpected format")
        
else:
    print(f"❌ VCC file not found: {vcc_file}")

# =============================================================================
# 4. COMMON VCC SUBMISSION ISSUES
# =============================================================================
print(f"\n🔍 CHECKING COMMON VCC ISSUES")
print("-" * 35)

issues_found = []
recommendations = []

# Issue 1: Gene count mismatch
expected_genes = 18080  # Based on your data
if adata is not None:
    actual_genes = adata.n_vars
    if actual_genes != expected_genes:
        issues_found.append(f"Gene count: expected {expected_genes}, got {actual_genes}")
        recommendations.append("Check if all required genes are present")

# Issue 2: Missing perturbations
if adata is not None and 'target_gene' in adata.obs.columns:
    pert_counts = adata.obs['target_gene'].value_counts()
    
    # Check validation perturbations count
    expected_perts = 51  # Based on your output
    actual_perts = len(pert_counts)
    if actual_perts != expected_perts:
        issues_found.append(f"Perturbation count: expected {expected_perts}, got {actual_perts}")
        recommendations.append("Verify all validation perturbations are included")
    
    # Check for non-targeting
    if 'non-targeting' not in pert_counts:
        issues_found.append("Missing 'non-targeting' control")
        recommendations.append("Ensure non-targeting control is properly named")

# Issue 3: Data format issues
if adata is not None:
    # Check for appropriate data range
    if hasattr(adata.X, 'toarray'):
        sample_data = adata.X[:100].toarray()
    else:
        sample_data = adata.X[:100]
    
    data_max = sample_data.max()
    data_mean = sample_data.mean()
    
    if data_max > 20:  # Suspiciously high for log-normalized data
        issues_found.append(f"Very high expression values: max = {data_max:.2f}")
        recommendations.append("Check if data is properly log-normalized")
    
    if data_mean > 5:  # High mean suggests raw counts
        issues_found.append(f"High mean expression: {data_mean:.2f}")
        recommendations.append("Data may not be properly normalized")

# Issue 4: File size issues
if os.path.exists(vcc_file):
    vcc_size = os.path.getsize(vcc_file) / 1e6
    if vcc_size > 2000:  # > 2GB
        issues_found.append(f"Very large VCC file: {vcc_size:.1f} MB")
        recommendations.append("Consider data compression or sparsification")

# Report issues
if issues_found:
    print("⚠️ POTENTIAL ISSUES DETECTED:")
    for i, issue in enumerate(issues_found, 1):
        print(f"   {i}. {issue}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
else:
    print("✅ No obvious issues detected")

# =============================================================================
# 5. GENERATE COMPARISON WITH WORKING SUBMISSION
# =============================================================================
print(f"\n📊 COMPARISON WITH OTHER SUBMISSIONS")
print("-" * 40)

# Look for other successful .vcc files
parent_dir = Path("..").resolve()
vcc_files = list(parent_dir.glob("*/*.prep.vcc"))

if vcc_files:
    print(f"Found {len(vcc_files)} other VCC files:")
    
    current_vcc_path = Path(vcc_file).resolve()
    
    for vcc_path in vcc_files:
        if vcc_path != current_vcc_path:
            rel_path = vcc_path.relative_to(parent_dir)
            size = vcc_path.stat().st_size / 1e6
            print(f"   📦 {rel_path} ({size:.1f} MB)")
            
            # Compare with current file
            current_size = os.path.getsize(vcc_file) / 1e6
            size_ratio = current_size / size
            
            if 0.8 <= size_ratio <= 1.2:  # Similar size
                print(f"       ✅ Similar size to current ({size_ratio:.2f}x)")
            elif size_ratio > 2:
                print(f"       ⚠️ Much larger than current ({size_ratio:.2f}x)")
            elif size_ratio < 0.5:
                print(f"       ⚠️ Much smaller than current ({size_ratio:.2f}x)")
else:
    print("No other VCC files found for comparison")

# =============================================================================
# 6. GENERATE DIAGNOSTIC REPORT
# =============================================================================
print(f"\n📄 GENERATING DIAGNOSTIC REPORT")
print("-" * 35)

report_path = "vcc_diagnostic_report.txt"

with open(report_path, 'w') as f:
    f.write("VCC SUBMISSION DIAGNOSTIC REPORT\n")
    f.write("=" * 40 + "\n\n")
    
    f.write("FILES ANALYZED:\n")
    f.write(f"- H5AD file: {h5ad_file}\n")
    f.write(f"- VCC file: {vcc_file}\n")
    f.write(f"- Gene names: {gene_names_file}\n\n")
    
    if adata is not None:
        f.write("H5AD FILE DETAILS:\n")
        f.write(f"- Shape: {adata.shape}\n")
        f.write(f"- Cells: {adata.n_obs:,}\n")
        f.write(f"- Genes: {adata.n_vars:,}\n")
        f.write(f"- Obs columns: {list(adata.obs.columns)}\n")
        f.write(f"- Var columns: {list(adata.var.columns)}\n\n")
        
        if 'target_gene' in adata.obs.columns:
            pert_counts = adata.obs['target_gene'].value_counts()
            f.write("PERTURBATIONS:\n")
            f.write(f"- Total perturbations: {len(pert_counts)}\n")
            f.write(f"- Non-targeting cells: {pert_counts.get('non-targeting', 0):,}\n")
            f.write("- Top 10 perturbations:\n")
            for pert, count in pert_counts.head(10).items():
                f.write(f"  {pert}: {count:,} cells\n")
            f.write("\n")
    
    if issues_found:
        f.write("ISSUES DETECTED:\n")
        for issue in issues_found:
            f.write(f"- {issue}\n")
        f.write("\n")
        
        f.write("RECOMMENDATIONS:\n")
        for rec in recommendations:
            f.write(f"- {rec}\n")
        f.write("\n")
    
    f.write("NEXT STEPS:\n")
    f.write("1. Review issues and recommendations above\n")
    f.write("2. Check VCC platform error messages\n")
    f.write("3. Compare with successful submissions\n")
    f.write("4. Consider regenerating files if major issues found\n")

print(f"✅ Diagnostic report saved: {report_path}")

# =============================================================================
# 7. SUGGESTED NEXT STEPS
# =============================================================================
print(f"\n🎯 SUGGESTED NEXT STEPS")
print("-" * 25)

print("1. 📋 Review the diagnostic report above")
print("2. 🔍 Check VCC platform for specific error messages")
print("3. 📊 Compare your submission with working examples")

if issues_found:
    print("4. 🔧 Fix identified issues:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
else:
    print("4. ✅ No obvious issues - problem may be platform-specific")

print("5. 💬 Share VCC platform error message for more specific help")
print("6. 🔄 Consider testing with smaller subset first")

print(f"\n📁 All diagnostic files saved in: {os.getcwd()}")
print(f"📄 Main report: {report_path}")

if adata is not None:
    del adata  # Clean up memory

print(f"\n❓ What specific error did you get from the VCC platform?")
print("This will help pinpoint the exact issue.")