#!/usr/bin/env python3
"""
Universal VCC Submission Prep Script
Finds and prepares any cross-dataset submission for VCC using cell-eval prep
"""

import pandas as pd
import anndata as ad
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import subprocess
import sys
import os
import glob


def find_submission_files():
    """Find the most recent submission files"""
    print("🔍 Finding submission files...")
    
    # Look for submission files in cross-dataset directory
    submission_dir = Path("vcc_cross_dataset_submission")
    
    if not submission_dir.exists():
        print(f"❌ Submission directory not found: {submission_dir}")
        return None, None
    
    # Possible submission file patterns
    submission_patterns = [
        "cross_dataset_full_submission*.h5ad",
        "cross_dataset_pseudobulk_submission*.h5ad", 
        "cross_dataset_submission*.h5ad",
        "cross_dataset_optimized*.h5ad",
        "cross_dataset_sparse*.h5ad",
        "*submission*.h5ad"
    ]
    
    submission_files = []
    for pattern in submission_patterns:
        files = list(submission_dir.glob(pattern))
        submission_files.extend(files)
    
    if not submission_files:
        print(f"❌ No submission files found in {submission_dir}")
        print("Available files:")
        for file in submission_dir.glob("*"):
            if file.is_file():
                print(f"   {file.name}")
        return None, None
    
    # Sort by modification time (most recent first)
    submission_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    submission_path = submission_files[0]
    gene_names_path = submission_dir / "gene_names.txt"
    
    print(f"✅ Found submission: {submission_path.name}")
    print(f"   Size: {submission_path.stat().st_size / 1e6:.1f} MB")
    print(f"   Modified: {pd.Timestamp.fromtimestamp(submission_path.stat().st_mtime)}")
    
    if not gene_names_path.exists():
        print(f"⚠️ Gene names file not found: {gene_names_path}")
        # Try to create it from the submission file
        try:
            adata = ad.read_h5ad(submission_path)
            with open(gene_names_path, 'w') as f:
                for gene in adata.var.index:
                    f.write(f"{gene}\n")
            print(f"✅ Created gene names file from submission")
        except Exception as e:
            print(f"❌ Could not create gene names file: {e}")
            return None, None
    
    return submission_path, gene_names_path


def optimize_submission(submission_path):
    """Optimize submission for cell-eval compatibility"""
    print(f"\n🔧 Optimizing submission for cell-eval...")
    
    # Load submission
    adata = ad.read_h5ad(submission_path)
    print(f"   Original shape: {adata.shape}")
    
    # Calculate memory requirements
    estimated_memory_gb = (adata.shape[0] * adata.shape[1] * 4) / 1e9
    print(f"   Estimated memory: {estimated_memory_gb:.2f} GB")
    
    optimized = False
    
    # Strategy 1: Optimize cell counts if too large
    if estimated_memory_gb > 3.0 or adata.shape[0] > 30000:
        print(f"   🎯 Reducing cell counts for memory efficiency...")
        
        target_genes = adata.obs['target_gene'].values
        unique_perts = np.unique(target_genes)
        
        # Conservative limits for cell-eval
        max_cells_per_pert = 300
        max_control_cells = 800
        
        selected_indices = []
        
        for pert in unique_perts:
            pert_mask = target_genes == pert
            pert_indices = np.where(pert_mask)[0]
            original_count = len(pert_indices)
            
            if pert == 'non-targeting':
                n_select = min(original_count, max_control_cells)
            else:
                n_select = min(original_count, max_cells_per_pert)
            
            if n_select > 0:
                if n_select < original_count:
                    selected = np.random.choice(pert_indices, n_select, replace=False)
                    selected_indices.extend(selected)
                    if len(unique_perts) <= 10:  # Only print for small number of perts
                        print(f"      {pert}: {original_count} → {n_select} cells")
                else:
                    selected_indices.extend(pert_indices)
        
        if len(selected_indices) < len(adata):
            selected_indices = sorted(selected_indices)
            adata_optimized = adata[selected_indices].copy()
            print(f"   ✅ Reduced to {len(selected_indices)} cells ({len(selected_indices)/len(adata)*100:.1f}% of original)")
            adata = adata_optimized
            optimized = True
    
    # Strategy 2: Convert to sparse if still large
    current_memory_gb = (adata.shape[0] * adata.shape[1] * 4) / 1e9
    if current_memory_gb > 2.0:
        print(f"   🗜️ Converting to sparse matrix...")
        
        # Get expression data
        X_dense = adata.X
        if hasattr(X_dense, 'toarray'):
            X_dense = X_dense.toarray()
        
        # Set low values to zero for sparsity
        threshold = 0.05
        X_dense[X_dense < threshold] = 0
        
        # Convert to sparse
        X_sparse = sp.csr_matrix(X_dense)
        adata.X = X_sparse
        
        sparsity = 1 - (X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]))
        print(f"   ✅ Sparsity: {sparsity*100:.1f}% zeros")
        optimized = True
    
    # Save optimized version if changes were made
    if optimized:
        optimized_path = submission_path.parent / f"{submission_path.stem}_optimized.h5ad"
        adata.write_h5ad(optimized_path, compression='gzip')
        print(f"   ✅ Saved optimized version: {optimized_path.name}")
        print(f"   ✅ Final shape: {adata.shape}")
        print(f"   ✅ Final size: {optimized_path.stat().st_size / 1e6:.1f} MB")
        return optimized_path
    else:
        print(f"   ✅ No optimization needed")
        return submission_path


def verify_gene_names(submission_path, gene_names_path):
    """Verify and fix gene names file"""
    print(f"\n📝 Verifying gene names...")
    
    # Load submission to get actual gene names
    adata = ad.read_h5ad(submission_path)
    expected_genes = list(adata.var.index)
    
    # Read gene names file
    with open(gene_names_path, 'r') as f:
        file_content = f.read().strip()
    
    file_genes = [g.strip() for g in file_content.split('\n') if g.strip()]
    
    print(f"   Expected genes: {len(expected_genes)}")
    print(f"   File genes: {len(file_genes)}")
    
    # Check for issues
    issues_found = False
    
    if len(file_genes) != len(expected_genes):
        print(f"   ⚠️ Gene count mismatch!")
        issues_found = True
    
    # Check for problematic gene names
    problematic_genes = []
    for gene in expected_genes:
        gene_str = str(gene).strip().lower()
        if gene_str in ['gene', 'nan', '', 'none', 'null'] or len(gene_str) < 2:
            problematic_genes.append(gene)
    
    if problematic_genes:
        print(f"   ⚠️ Found {len(problematic_genes)} problematic genes:")
        for gene in problematic_genes[:5]:  # Show first 5
            print(f"      '{gene}'")
        if len(problematic_genes) > 5:
            print(f"      ... and {len(problematic_genes) - 5} more")
        issues_found = True
    
    # Fix gene names file if needed
    if issues_found:
        print(f"   🔧 Fixing gene names file...")
        
        # Clean gene names
        cleaned_genes = []
        for gene in expected_genes:
            gene_str = str(gene).strip()
            if gene_str.lower() in ['gene', 'nan', '', 'none', 'null'] or len(gene_str) < 2:
                # Replace problematic genes with a valid placeholder
                cleaned_genes.append(f"GENE_{len(cleaned_genes):05d}")
            else:
                cleaned_genes.append(gene_str)
        
        # Write cleaned gene names
        with open(gene_names_path, 'w') as f:
            for gene in cleaned_genes:
                f.write(f"{gene}\n")
        
        print(f"   ✅ Fixed gene names file: {len(cleaned_genes)} genes")
        
        # Also update the submission file with cleaned gene names
        adata.var.index = cleaned_genes
        adata.write_h5ad(submission_path, compression='gzip')
        print(f"   ✅ Updated submission file with cleaned gene names")
    else:
        print(f"   ✅ Gene names file is correct")
    
    return True


def run_cell_eval_prep(submission_path, gene_names_path):
    """Run cell-eval prep on the submission"""
    print(f"\n📦 Running cell-eval prep...")
    print(f"   Input: {submission_path.name}")
    print(f"   Genes: {gene_names_path.name}")
    
    cmd = [
        sys.executable, "-m", "cell_eval", "prep",
        "-i", str(submission_path),
        "--genes", str(gene_names_path)
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        # Run with timeout
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True, 
            timeout=1200,  # 20 minute timeout
            cwd=submission_path.parent
        )
        
        print("✅ cell-eval prep completed successfully!")
        
        if result.stdout:
            print("\n📋 Output:")
            # Show last few lines of output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-10:]:
                if line.strip():
                    print(f"   {line}")
        
        # Find the output .prep.vcc file
        possible_outputs = [
            submission_path.with_suffix('.prep.vcc'),
            submission_path.parent / f"{submission_path.stem}.prep.vcc"
        ]
        
        output_file = None
        for output_path in possible_outputs:
            if output_path.exists():
                output_file = output_path
                break
        
        if output_file:
            final_size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"\n🎉 SUCCESS!")
            print(f"✅ VCC submission ready: {output_file.name}")
            print(f"✅ File size: {final_size_mb:.1f} MB")
            return str(output_file)
        else:
            print("❌ No .prep.vcc output file found")
            print("Available files after prep:")
            for file in submission_path.parent.glob("*"):
                if file.is_file():
                    print(f"   {file.name}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ cell-eval prep failed with return code {e.returncode}")
        if e.stderr:
            print(f"Error output:")
            for line in e.stderr.strip().split('\n'):
                if line.strip():
                    print(f"   {line}")
        if e.stdout:
            print(f"Standard output:")
            for line in e.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line}")
        
        print(f"\n💾 Original H5AD file is still valid: {submission_path.name}")
        return str(submission_path)
        
    except subprocess.TimeoutExpired:
        print("❌ cell-eval prep timed out (20 minutes)")
        print("This may indicate memory issues or very large dataset")
        return str(submission_path)
    
    except FileNotFoundError:
        print("❌ cell-eval not found!")
        print("Please install cell-eval: pip install cell-eval")
        return None


def create_submission_summary(final_file, submission_path):
    """Create a summary of the submission"""
    print(f"\n📋 Creating submission summary...")
    
    # Load final submission for stats
    try:
        adata = ad.read_h5ad(submission_path)
        
        summary_path = submission_path.parent / "submission_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("VCC CROSS-DATASET SUBMISSION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("SUBMISSION FILES:\n")
            f.write("-" * 16 + "\n")
            if final_file.endswith('.prep.vcc'):
                f.write(f"✅ VCC Ready File: {Path(final_file).name}\n")
                f.write(f"✅ Upload Status: Ready for VCC platform\n")
            else:
                f.write(f"📊 H5AD File: {Path(final_file).name}\n")
                f.write(f"⚠️ Upload Status: May need manual prep\n")
            
            f.write(f"\nDATA STATISTICS:\n")
            f.write("-" * 16 + "\n")
            f.write(f"Total Cells: {adata.shape[0]:,}\n")
            f.write(f"Total Genes: {adata.shape[1]:,}\n")
            f.write(f"Perturbations: {adata.obs['target_gene'].nunique()}\n")
            
            pert_counts = adata.obs['target_gene'].value_counts()
            f.write(f"Non-targeting cells: {pert_counts.get('non-targeting', 0):,}\n")
            
            f.write(f"\nFILE INFORMATION:\n")
            f.write("-" * 16 + "\n")
            final_size = Path(final_file).stat().st_size / (1024 * 1024)
            f.write(f"File Size: {final_size:.1f} MB\n")
            f.write(f"Compression: {'Sparse' if hasattr(adata.X, 'nnz') else 'Dense'}\n")
            
            if hasattr(adata.X, 'nnz'):
                sparsity = 1 - (adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]))
                f.write(f"Sparsity: {sparsity*100:.1f}% zeros\n")
            
            f.write(f"\nMODEL INFORMATION:\n")
            f.write("-" * 17 + "\n")
            f.write(f"Approach: Cross-Dataset Random Forest\n")
            f.write(f"Features: Gene Expression + Embeddings\n")
            f.write(f"Training: Multiple datasets for robustness\n")
            f.write(f"Prediction: Differential expression modeling\n")
            
            f.write(f"\nNEXT STEPS:\n")
            f.write("-" * 11 + "\n")
            if final_file.endswith('.prep.vcc'):
                f.write(f"1. Upload {Path(final_file).name} to VCC platform\n")
                f.write(f"2. Monitor submission status\n")
                f.write(f"3. Compare results with other approaches\n")
            else:
                f.write(f"1. Try uploading {Path(final_file).name} directly to VCC\n")
                f.write(f"2. If upload fails, contact VCC support\n")
                f.write(f"3. Consider further optimization if needed\n")
        
        print(f"✅ Summary saved: {summary_path.name}")
        
    except Exception as e:
        print(f"⚠️ Could not create summary: {e}")


def main():
    """Main function to prepare VCC submission"""
    print("🧬 VCC SUBMISSION PREPARATION")
    print("=" * 50)
    
    try:
        # Find submission files
        submission_path, gene_names_path = find_submission_files()
        if not submission_path:
            return 1
        
        # Optimize submission for cell-eval
        optimized_path = optimize_submission(submission_path)
        
        # Verify and fix gene names
        verify_gene_names(optimized_path, gene_names_path)
        
        # Run cell-eval prep
        final_file = run_cell_eval_prep(optimized_path, gene_names_path)
        
        if final_file:
            # Create summary
            create_submission_summary(final_file, optimized_path)
            
            print(f"\n🎉 VCC SUBMISSION PREPARATION COMPLETE!")
            print("=" * 60)
            
            if final_file.endswith('.prep.vcc'):
                print(f"✅ Ready-to-upload file: {Path(final_file).name}")
                print(f"🚀 Upload this file to the Virtual Cell Challenge platform!")
                
                print(f"\n💡 SUBMISSION ADVANTAGES:")
                print(f"   🏆 Cross-dataset training for robustness")
                print(f"   🏆 Full gene coverage (~18k genes)")
                print(f"   🏆 Optimized for VCC platform")
                print(f"   🏆 Memory-efficient format")
            else:
                print(f"✅ Submission file: {Path(final_file).name}")
                print(f"⚠️ Try uploading directly or contact VCC support if needed")
            
            print(f"\n📁 Location: vcc_cross_dataset_submission/")
            return 0
        else:
            print(f"\n❌ Failed to prepare VCC submission")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during preparation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())