#!/usr/bin/env python3
"""
Check perturbation names in training vs prediction data
"""

import pandas as pd
import anndata as ad
from pathlib import Path

def check_perturbation_names():
    """Compare perturbation names between datasets"""
    
    print("🔍 CHECKING PERTURBATION NAMES")
    print("=" * 40)
    
    # Load datasets
    true_data_path = "data/raw/single_cell_rnaseq/vcc_data/adata_Training.h5ad"
    pred_data_path = "vcc_esm2_rf_submission/esm2_rf_submission.h5ad"
    
    print("📥 Loading training data...")
    true_data = ad.read_h5ad(true_data_path)
    true_perts = set(true_data.obs['target_gene'].unique())
    
    print("📥 Loading prediction data...")
    pred_data = ad.read_h5ad(pred_data_path)
    pred_perts = set(pred_data.obs['target_gene'].unique())
    
    print(f"\n📊 PERTURBATION COMPARISON:")
    print(f"   Training data: {len(true_perts)} unique perturbations")
    print(f"   Prediction data: {len(pred_perts)} unique perturbations")
    
    # Find overlaps
    common = true_perts.intersection(pred_perts)
    true_only = true_perts - pred_perts
    pred_only = pred_perts - true_perts
    
    print(f"   Common: {len(common)}")
    print(f"   Training only: {len(true_only)}")
    print(f"   Prediction only: {len(pred_only)}")
    
    print(f"\n✅ COMMON PERTURBATIONS ({len(common)}):")
    for pert in sorted(list(common)):
        true_count = sum(true_data.obs['target_gene'] == pert)
        pred_count = sum(pred_data.obs['target_gene'] == pert)
        print(f"   {pert}: {true_count} (true) vs {pred_count} (pred) cells")
    
    if len(true_only) > 0:
        print(f"\n⚠️ TRAINING ONLY ({len(true_only)}):")
        for pert in sorted(list(true_only))[:10]:
            count = sum(true_data.obs['target_gene'] == pert)
            print(f"   {pert}: {count} cells")
        if len(true_only) > 10:
            print(f"   ... and {len(true_only) - 10} more")
    
    if len(pred_only) > 0:
        print(f"\n⚠️ PREDICTION ONLY ({len(pred_only)}):")
        for pert in sorted(list(pred_only))[:10]:
            count = sum(pred_data.obs['target_gene'] == pert)
            print(f"   {pert}: {count} cells")
        if len(pred_only) > 10:
            print(f"   ... and {len(pred_only) - 10} more")
    
    # Check validation set
    pert_counts_path = Path("data/raw/single_cell_rnaseq/vcc_data/pert_counts_Validation.csv")
    if pert_counts_path.exists():
        print(f"\n📋 VALIDATION SET CHECK:")
        pert_counts = pd.read_csv(pert_counts_path)
        val_perts = set(pert_counts['target_gene'].unique())
        
        val_in_true = val_perts.intersection(true_perts)
        val_in_pred = val_perts.intersection(pred_perts)
        val_common = val_in_true.intersection(val_in_pred)
        
        print(f"   Validation perturbations: {len(val_perts)}")
        print(f"   In training data: {len(val_in_true)}")
        print(f"   In prediction data: {len(val_in_pred)}")
        print(f"   Available for evaluation: {len(val_common)}")
        
        if len(val_common) > 0:
            print(f"\n✅ VALIDATION PERTURBATIONS FOR EVALUATION:")
            for pert in sorted(list(val_common)):
                val_target = pert_counts[pert_counts['target_gene'] == pert]['n_cells'].iloc[0]
                true_count = sum(true_data.obs['target_gene'] == pert)
                pred_count = sum(pred_data.obs['target_gene'] == pert)
                print(f"   {pert}: target={val_target}, true={true_count}, pred={pred_count}")
    
    return common

if __name__ == "__main__":
    common_perts = check_perturbation_names()
    
    if len(common_perts) > 3:
        print(f"\n🎉 EVALUATION IS POSSIBLE!")
        print(f"✅ {len(common_perts)} common perturbations found")
        print(f"🚀 Run vcc_evaluation_viz.py to proceed")
    else:
        print(f"\n❌ EVALUATION PROBLEMATIC!")
        print(f"⚠️ Only {len(common_perts)} common perturbations")
        print(f"💡 Check if perturbation names match between datasets")