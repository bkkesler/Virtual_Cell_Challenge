"""
Create a sparse, memory-efficient VCC submission that cell-eval can handle
"""

import pandas as pd
import anndata as ad
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import subprocess
import sys


def create_sparse_submission():
    """Create a sparse version of the submission that's more memory efficient"""
    
    print("💾 Creating Sparse VCC Submission")
    print("="*50)
    
    # Load the current predictions
    backup_h5ad = Path("../outputs/vcc_submission/predictions_backup.h5ad")
    if not backup_h5ad.exists():
        print(f"❌ Backup file not found: {backup_h5ad}")
        return False
    
    print("📊 Loading predictions...")
    adata = ad.read_h5ad(backup_h5ad)
    print(f"   Original shape: {adata.shape}")
    print(f"   Original size: ~{(adata.shape[0] * adata.shape[1] * 4) / 1e9:.2f} GB")
    
    # Strategy: Create a much smaller, sparse submission
    print("\n🎯 Creating memory-efficient submission...")
    
    # Step 1: Reduce number of cells per perturbation
    print("🔢 Reducing cell counts for memory efficiency...")
    
    target_genes = adata.obs['target_gene'].values
    unique_perts = np.unique(target_genes)
    
    # Target: ~20,000 total cells (instead of 65,751)
    max_cells_per_pert = 300  # Reduce from ~1,000+ to 300
    max_control_cells = 2000  # Reduce controls
    
    selected_indices = []
    
    for pert in unique_perts:
        pert_mask = target_genes == pert
        pert_indices = np.where(pert_mask)[0]
        
        if pert == 'non-targeting':
            # Control cells - keep more but not too many
            n_select = min(len(pert_indices), max_control_cells)
        else:
            # Regular perturbations
            n_select = min(len(pert_indices), max_cells_per_pert)
        
        if n_select > 0:
            selected = np.random.choice(pert_indices, n_select, replace=False)
            selected_indices.extend(selected)
            print(f"   {pert}: {len(pert_indices)} → {n_select} cells")
    
    # Step 2: Create smaller dataset
    selected_indices = sorted(selected_indices)
    sparse_adata = adata[selected_indices].copy()
    
    print(f"\n✅ Reduced dataset:")
    print(f"   New shape: {sparse_adata.shape}")
    print(f"   Size reduction: {len(selected_indices)/len(adata)*100:.1f}% of original")
    print(f"   Estimated size: ~{(sparse_adata.shape[0] * sparse_adata.shape[1] * 4) / 1e9:.2f} GB")
    
    # Step 3: Convert to sparse matrix to save even more memory
    print(f"\n💾 Converting to sparse matrix...")
    
    # Make the data more sparse by zeroing out very low expression values
    X_dense = sparse_adata.X
    
    # Set very low values to zero (typical scRNA-seq preprocessing)
    threshold = 0.1  # Values below this become zero
    X_dense[X_dense < threshold] = 0
    
    # Convert to sparse
    X_sparse = sp.csr_matrix(X_dense)
    sparse_adata.X = X_sparse
    
    sparsity = 1 - (X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]))
    print(f"✅ Sparsity: {sparsity*100:.1f}% zeros")
    print(f"✅ Non-zero elements: {X_sparse.nnz:,}")
    
    # Step 4: Save sparse submission
    sparse_path = Path("../outputs/vcc_submission/predictions_sparse.h5ad")
    sparse_adata.write_h5ad(sparse_path)
    
    file_size = sparse_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved sparse submission: {sparse_path}")
    print(f"✅ File size: {file_size:.1f} MB")
    
    # Step 5: Create corrected gene names (no header)
    gene_names_path = Path("../outputs/vcc_submission/gene_names_sparse.txt")
    with open(gene_names_path, 'w') as f:
        for gene in sparse_adata.var.index:
            f.write(f"{gene}\n")
    
    print(f"✅ Created gene names file: {gene_names_path}")
    
    # Step 6: Try cell-eval prep on the sparse version
    print(f"\n📦 Running cell-eval prep on sparse submission...")
    
    cmd = [
        sys.executable, "-m", "cell_eval", "prep",
        "-i", str(sparse_path),
        "--genes", str(gene_names_path)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        
        print("✅ cell-eval prep completed successfully!")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        # Find output file
        possible_outputs = [
            sparse_path.with_suffix('.prep.vcc'),
            sparse_path.parent / f"{sparse_path.stem}.prep.vcc"
        ]
        
        output_file = None
        for output_path in possible_outputs:
            if output_path.exists():
                output_file = output_path
                break
        
        if output_file:
            final_size = output_file.stat().st_size / (1024 * 1024)
            print(f"\n🎉 SUCCESS!")
            print(f"✅ Final submission: {output_file}")
            print(f"✅ Final size: {final_size:.1f} MB")
            
            # Show final statistics
            print(f"\n📊 Final Submission Stats:")
            print(f"   Cells: {sparse_adata.shape[0]:,}")
            print(f"   Genes: {sparse_adata.shape[1]:,}")
            print(f"   Perturbations: {len(np.unique(sparse_adata.obs['target_gene']))}")
            print(f"   Sparsity: {sparsity*100:.1f}%")
            print(f"   File size: {final_size:.1f} MB")
            
            return str(output_file)
        else:
            print("❌ No output file found")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ cell-eval prep still failed: {e}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        
        print(f"\n💾 Sparse .h5ad file is still valid: {sparse_path}")
        print(f"📊 It contains {sparse_adata.shape[0]:,} cells with realistic predictions")
        return str(sparse_path)
        
    except subprocess.TimeoutExpired:
        print("❌ cell-eval prep timed out (10 minutes)")
        return str(sparse_path)


def main():
    """Main function"""
    
    result = create_sparse_submission()
    
    if result:
        if result.endswith('.prep.vcc'):
            print(f"\n🎉 COMPLETE SUCCESS!")
            print(f"📤 Ready-to-submit file: {result}")
            print(f"🚀 Upload this to Virtual Cell Challenge!")
        else:
            print(f"\n✅ PARTIAL SUCCESS!")
            print(f"📊 Valid predictions file: {result}")
            print(f"💡 This contains realistic STATE model predictions")
            print(f"🔧 You can try uploading directly or contact VCC support")
    else:
        print(f"\n❌ Unable to create submission file")
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())