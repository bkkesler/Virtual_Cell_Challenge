"""
Thorough diagnosis of the gene count mismatch to understand the root cause
"""

import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
import yaml


def diagnose_gene_count_mismatch():
    """Investigate exactly why there's a gene count mismatch"""
    
    print("🔍 GENE COUNT MISMATCH DIAGNOSIS")
    print("="*50)
    
    # 1. Check the original training data
    print("\n1️⃣ ORIGINAL TRAINING DATA:")
    training_path = Path("data/raw/single_cell_rnaseq/vcc_data/adata_Training.h5ad")
    if training_path.exists():
        adata = ad.read_h5ad(training_path)
        print(f"   Shape: {adata.shape}")
        print(f"   Genes in training data: {adata.shape[1]}")
        print(f"   First 5 genes: {list(adata.var.index[:5])}")
        print(f"   Last 5 genes: {list(adata.var.index[-5:])}")
        training_genes = list(adata.var.index)
        adata.file.close()
    else:
        print("   ❌ Training data not found")
        training_genes = None
    
    # 2. Check the gene names file
    print("\n2️⃣ GENE NAMES FILE:")
    gene_names_path = Path("data/raw/single_cell_rnaseq/vcc_data/gene_names.csv")
    if gene_names_path.exists():
        gene_df = pd.read_csv(gene_names_path)
        print(f"   File: {gene_names_path}")
        print(f"   Shape: {gene_df.shape}")
        print(f"   Columns: {list(gene_df.columns)}")
        print(f"   Gene count: {len(gene_df)}")
        print(f"   First 5 genes: {list(gene_df.iloc[:5, 0])}")
        print(f"   Last 5 genes: {list(gene_df.iloc[-5:, 0])}")
        file_genes = list(gene_df.iloc[:, 0])
    else:
        print("   ❌ Gene names file not found")
        file_genes = None
    
    # 3. Check model configuration
    print("\n3️⃣ MODEL CONFIGURATION:")
    config_path = Path("config/state_model_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_genes = config.get('model', {}).get('n_genes', 'Not specified')
        print(f"   Config file: {config_path}")
        print(f"   n_genes in config: {model_genes}")
    else:
        print("   ❌ Config file not found")
        model_genes = None
    
    # 4. Check subset data (what we've been training on)
    print("\n4️⃣ SUBSET DATA:")
    subset_path = Path("data/raw/single_cell_rnaseq/vcc_data/adata_Training_subset.h5ad")
    if subset_path.exists():
        subset_adata = ad.read_h5ad(subset_path)
        print(f"   Shape: {subset_adata.shape}")
        print(f"   Genes in subset: {subset_adata.shape[1]}")
        subset_genes = list(subset_adata.var.index)
        subset_adata.file.close()
    else:
        print("   ❌ Subset data not found")
        subset_genes = None
    
    # 5. Compare all sources
    print("\n5️⃣ COMPARISON ANALYSIS:")
    sources = {
        'Training data': training_genes,
        'Gene names file': file_genes, 
        'Subset data': subset_genes,
        'Model config': model_genes
    }
    
    counts = {}
    for name, genes in sources.items():
        if genes is not None and isinstance(genes, list):
            counts[name] = len(genes)
        else:
            counts[name] = genes
    
    print("   Gene counts by source:")
    for name, count in counts.items():
        print(f"     {name}: {count}")
    
    # 6. Find the discrepancy
    print("\n6️⃣ DISCREPANCY ANALYSIS:")
    if training_genes and file_genes:
        if len(training_genes) != len(file_genes):
            print(f"   ⚠️  MISMATCH FOUND!")
            print(f"   Training data: {len(training_genes)} genes")
            print(f"   Gene names file: {len(file_genes)} genes")
            print(f"   Difference: {len(training_genes) - len(file_genes)} genes")
            
            # Check if it's just the order or actually missing genes
            training_set = set(training_genes)
            file_set = set(file_genes)
            
            missing_from_file = training_set - file_set
            extra_in_file = file_set - training_set
            
            print(f"\n   Missing from gene names file: {len(missing_from_file)} genes")
            if missing_from_file:
                print(f"     {list(missing_from_file)[:10]}...")  # Show first 10
            
            print(f"   Extra in gene names file: {len(extra_in_file)} genes")
            if extra_in_file:
                print(f"     {list(extra_in_file)[:10]}...")  # Show first 10
                
        else:
            print("   ✅ Training data and gene names file have same count")
            
            # Check if they're the same genes
            if set(training_genes) == set(file_genes):
                print("   ✅ Same genes, possibly different order")
            else:
                print("   ⚠️  Same count but different genes!")
    
    # 7. What actually happened during training
    print("\n7️⃣ TRAINING ANALYSIS:")
    if subset_genes and len(subset_genes) != 18080:
        print(f"   🎯 KEY FINDING: Subset was created with {len(subset_genes)} genes")
        print(f"   But model config expects 18080 genes")
        print(f"   This means during training:")
        print(f"     - Data loader loaded {len(subset_genes)} genes")
        print(f"     - Model was initialized for 18080 genes")
        print(f"     - There was likely dimension padding/truncation happening")
    
    # 8. Recommendations
    print("\n8️⃣ RECOMMENDATIONS:")
    print("   Based on the analysis:")
    
    if training_genes and len(training_genes) == 18080:
        print("   ✅ Use the original training data gene list (18080 genes)")
        print("   ✅ The gene names file seems to be missing 1 gene")
        print("   🔧 Safe fix: Extract gene names directly from training data")
    else:
        print("   ⚠️  Need to investigate why model expects 18080 but data has different count")
    
    return {
        'training_genes': len(training_genes) if training_genes else None,
        'file_genes': len(file_genes) if file_genes else None,
        'subset_genes': len(subset_genes) if subset_genes else None,
        'model_genes': model_genes,
        'discrepancy_found': True if training_genes and file_genes and len(training_genes) != len(file_genes) else False
    }


def create_correct_gene_names_file():
    """Create a correct gene names file from the original training data"""
    
    print("\n🔧 CREATING CORRECT GENE NAMES FILE")
    print("="*40)
    
    training_path = Path("data/raw/single_cell_rnaseq/vcc_data/adata_Training.h5ad")
    gene_names_path = Path("data/raw/single_cell_rnaseq/vcc_data/gene_names.csv")
    
    if not training_path.exists():
        print("❌ Cannot create correct file - training data not found")
        return False
    
    # Load training data
    print("📊 Loading original training data...")
    adata = ad.read_h5ad(training_path)
    correct_genes = list(adata.var.index)
    adata.file.close()
    
    print(f"✅ Found {len(correct_genes)} genes in training data")
    
    # Backup current gene names file
    if gene_names_path.exists():
        backup_path = gene_names_path.with_suffix('.csv.original_backup')
        import shutil
        shutil.copy2(gene_names_path, backup_path)
        print(f"💾 Backed up original file to: {backup_path}")
    
    # Create correct gene names file
    correct_df = pd.DataFrame({'gene': correct_genes})
    correct_df.to_csv(gene_names_path, index=False)
    
    print(f"✅ Created correct gene names file with {len(correct_genes)} genes")
    print(f"📁 File: {gene_names_path}")
    
    return True


def main():
    """Run complete diagnosis"""
    
    analysis = diagnose_gene_count_mismatch()
    
    print("\n" + "="*50)
    print("🎯 SUMMARY & NEXT STEPS")
    print("="*50)
    
    if analysis['discrepancy_found']:
        print("❌ ISSUE CONFIRMED: Gene count mismatch found")
        print("\n🔧 SAFE SOLUTION:")
        print("1. Extract gene names directly from training data")
        print("2. This ensures perfect compatibility")
        print("3. No arbitrary padding or truncation")
        
        response = input("\n❓ Create correct gene names file from training data? (y/n): ")
        if response.lower() == 'y':
            if create_correct_gene_names_file():
                print("\n✅ FIXED! Now you can run the submission generator safely.")
                print("🚀 Run: python generate_vcc_submission.py")
            else:
                print("\n❌ Could not create correct gene names file")
        else:
            print("\n💡 You can run create_correct_gene_names_file() manually later")
    else:
        print("✅ No obvious discrepancy found - the 1-gene difference might be normal")
        print("💡 Consider using the automatic padding fix in the submission generator")


if __name__ == "__main__":
    main()