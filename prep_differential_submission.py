"""
Prepare the differential ESM2 Random Forest submission for VCC using cell-eval prep
"""

import pandas as pd
import anndata as ad
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import subprocess
import sys
import os


def prep_differential_submission():
    """Prepare the differential submission for VCC"""
    
    print("📦 PREPARING DIFFERENTIAL ESM2 SUBMISSION FOR VCC")
    print("=" * 60)
    
    # Load the differential predictions
    submission_path = Path("vcc_esm2_rf_differential_submission/esm2_rf_differential_submission.h5ad")
    gene_names_path = Path("vcc_esm2_rf_differential_submission/gene_names.txt")
    
    if not submission_path.exists():
        print(f"❌ Submission file not found: {submission_path}")
        print("Please run the differential model script first.")
        return False
        
    if not gene_names_path.exists():
        print(f"❌ Gene names file not found: {gene_names_path}")
        return False
    
    print("📊 Loading differential predictions...")
    adata = ad.read_h5ad(submission_path)
    print(f"   Shape: {adata.shape}")
    print(f"   Size: ~{(adata.shape[0] * adata.shape[1] * 4) / 1e9:.2f} GB")
    
    # Check if we need to make it more memory efficient
    estimated_memory_gb = (adata.shape[0] * adata.shape[1] * 4) / 1e9
    
    if estimated_memory_gb > 4.0:  # If larger than 4GB, make it more efficient
        print(f"\n💾 Creating memory-efficient version...")
        print(f"🎯 Current size ({estimated_memory_gb:.1f} GB) may cause cell-eval issues")
        
        # Strategy 1: Reduce cells per perturbation if needed
        target_genes = adata.obs['target_gene'].values
        unique_perts = np.unique(target_genes)
        
        # Target: ~25,000 total cells maximum
        max_cells_per_pert = 400
        max_control_cells = 1500
        
        selected_indices = []
        
        print("🔢 Optimizing cell counts for cell-eval...")
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
                    print(f"   {pert}: {original_count} → {n_select} cells")
                else:
                    selected_indices.extend(pert_indices)
                    print(f"   {pert}: {original_count} cells (kept all)")
        
        # Create optimized dataset
        selected_indices = sorted(selected_indices)
        adata_optimized = adata[selected_indices].copy()
        
        print(f"\n✅ Optimized dataset:")
        print(f"   New shape: {adata_optimized.shape}")
        print(f"   Size reduction: {len(selected_indices)/len(adata)*100:.1f}% of original")
        print(f"   Estimated size: ~{(adata_optimized.shape[0] * adata_optimized.shape[1] * 4) / 1e9:.2f} GB")
        
        # Use optimized version
        adata = adata_optimized
        
        # Save optimized version
        optimized_path = submission_path.parent / "esm2_rf_differential_optimized.h5ad"
        adata.write_h5ad(optimized_path)
        print(f"✅ Saved optimized version: {optimized_path}")
        submission_path = optimized_path
    
    # Strategy 2: Make data sparser if still large
    current_size_gb = (adata.shape[0] * adata.shape[1] * 4) / 1e9
    if current_size_gb > 3.0:
        print(f"\n💾 Converting to sparse matrix for efficiency...")
        
        # Get expression data
        X_dense = adata.X
        if hasattr(X_dense, 'toarray'):
            X_dense = X_dense.toarray()
        
        # Set very low values to zero (typical scRNA-seq preprocessing)
        threshold = 0.05  # Conservative threshold for differential data
        X_dense[X_dense < threshold] = 0
        
        # Convert to sparse
        X_sparse = sp.csr_matrix(X_dense)
        adata.X = X_sparse
        
        sparsity = 1 - (X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]))
        print(f"✅ Sparsity: {sparsity*100:.1f}% zeros")
        print(f"✅ Non-zero elements: {X_sparse.nnz:,}")
        
        # Save sparse version
        sparse_path = submission_path.parent / "esm2_rf_differential_sparse.h5ad"
        adata.write_h5ad(sparse_path)
        print(f"✅ Saved sparse version: {sparse_path}")
        submission_path = sparse_path
    
    # Ensure gene names file is correct
    print(f"\n📝 Verifying gene names file...")
    with open(gene_names_path, 'r') as f:
        gene_names_content = f.read().strip()
    
    # Check if gene names match and have no header
    expected_genes = adata.var.index.tolist()
    file_genes = gene_names_content.split('\n')
    
    if len(file_genes) != len(expected_genes):
        print(f"⚠️ Gene count mismatch. Fixing gene names file...")
        with open(gene_names_path, 'w') as f:
            for gene in expected_genes:
                f.write(f"{gene}\n")
        print(f"✅ Fixed gene names file: {len(expected_genes)} genes")
    else:
        print(f"✅ Gene names file verified: {len(file_genes)} genes")
    
    # Run cell-eval prep
    print(f"\n📦 Running cell-eval prep on differential submission...")
    print(f"📁 Input: {submission_path}")
    print(f"📝 Genes: {gene_names_path}")
    
    cmd = [
        sys.executable, "-m", "cell_eval", "prep",
        "-i", str(submission_path),
        "--genes", str(gene_names_path)
    ]
    
    print(f"🚀 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=900)  # 15 min timeout
        
        print("✅ cell-eval prep completed successfully!")
        
        if result.stdout:
            print("\n📋 STDOUT:")
            print(result.stdout)
        
        # Find output .prep.vcc file
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
            print(f"✅ Final VCC submission: {output_file}")
            print(f"✅ Final size: {final_size_mb:.1f} MB")
            
            # Show final statistics
            print(f"\n📊 Final Submission Stats:")
            print(f"   📊 Cells: {adata.shape[0]:,}")
            print(f"   🧬 Genes: {adata.shape[1]:,}")
            print(f"   🎯 Perturbations: {len(np.unique(adata.obs['target_gene']))}")
            print(f"   🤖 Model: ESM2 Differential Random Forest")
            print(f"   📁 File size: {final_size_mb:.1f} MB")
            
            return str(output_file)
        else:
            print("❌ No .prep.vcc output file found")
            print("📁 Available files:")
            for file in submission_path.parent.glob("*"):
                if file.is_file():
                    print(f"   {file}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ cell-eval prep failed: {e}")
        if e.stderr:
            print(f"📋 STDERR:")
            print(e.stderr)
        if e.stdout:
            print(f"📋 STDOUT:")
            print(e.stdout)
        
        print(f"\n💾 H5AD file is still valid: {submission_path}")
        print(f"📊 It contains {adata.shape[0]:,} cells with differential predictions")
        print(f"🔧 You can try uploading the .h5ad directly or contact VCC support")
        return str(submission_path)
        
    except subprocess.TimeoutExpired:
        print("❌ cell-eval prep timed out (15 minutes)")
        print(f"💾 This may indicate memory issues or very large dataset")
        return str(submission_path)
    
    except FileNotFoundError:
        print("❌ cell-eval not found!")
        print("📦 Please install cell-eval: pip install cell-eval")
        return None


def main():
    """Main function"""
    
    print("🧬 ESM2 DIFFERENTIAL RANDOM FOREST - VCC PREP")
    print("=" * 50)
    
    result = prep_differential_submission()
    
    if result:
        if result.endswith('.prep.vcc'):
            print(f"\n🎉 COMPLETE SUCCESS!")
            print(f"📤 Ready-to-submit file: {result}")
            print(f"🚀 Upload this .prep.vcc file to Virtual Cell Challenge!")
            print(f"\n💡 DIFFERENTIAL MODEL ADVANTAGES:")
            print(f"   🎯 Models perturbation effects directly")
            print(f"   📈 More biologically interpretable")
            print(f"   🔬 Better generalization potential")
            print(f"   ✨ Cleaner perturbation signatures")
        else:
            print(f"\n✅ PARTIAL SUCCESS!")
            print(f"📊 Valid predictions file: {result}")
            print(f"💡 Contains differential ESM2 Random Forest predictions")
            print(f"🔧 You can try uploading directly or contact VCC support")
    else:
        print(f"\n❌ Unable to create VCC submission file")
        print(f"💡 Check that cell-eval is installed and try again")
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())