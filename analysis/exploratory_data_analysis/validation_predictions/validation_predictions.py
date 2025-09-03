"""
Generate predictions for existing perturbations in training data using the trained Random Forest model.
This will be used for visualization and validation against real data.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import pickle
import joblib
import gc
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

print("🔬 VALIDATION PREDICTIONS GENERATOR")
print("=" * 50)
print("Generating predictions for existing perturbations using trained RF model")

# Configuration
MODEL_DIR = "./saved_esm2_rf_models"
OUTPUT_DIR = "./outputs/validation_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Settings
MIN_CELLS_PER_PERT = 50  # Minimum cells needed for meaningful comparison
MAX_CELLS_PER_PERT = 500  # Limit for memory efficiency
INCLUDE_CONTROLS = True  # Whether to include non-targeting controls

# =============================================================================
# STEP 1: LOAD TRAINED MODEL AND DATA
# =============================================================================
print("\n📁 LOADING TRAINED MODEL")
print("-" * 30)

# Load the trained Random Forest model
rf_model_path = os.path.join(MODEL_DIR, "esm2_rf_model.joblib")
control_mean_path = os.path.join(MODEL_DIR, "esm2_control_mean.npy")
esm2_embeddings_path = os.path.join(MODEL_DIR, "esm2_embeddings_dict.pkl")

if not all([os.path.exists(p) for p in [rf_model_path, control_mean_path, esm2_embeddings_path]]):
    print("❌ Required model files not found!")
    print(f"   RF Model: {os.path.exists(rf_model_path)}")
    print(f"   Control Mean: {os.path.exists(control_mean_path)}")
    print(f"   ESM2 Embeddings: {os.path.exists(esm2_embeddings_path)}")
    print("\nPlease train the model first using the ESM2 Random Forest script.")
    sys.exit(1)

print("📦 Loading trained Random Forest model...")
rf_model = joblib.load(rf_model_path)
control_mean = np.load(control_mean_path)

print("🧬 Loading ESM2 embeddings...")
with open(esm2_embeddings_path, 'rb') as f:
    esm2_embeddings = pickle.load(f)

print(f"✅ Model loaded successfully")
print(f"✅ Control baseline shape: {control_mean.shape}")
print(f"✅ ESM2 embeddings for {len(esm2_embeddings)} genes")

# Load training data
vcc_data_dir = r"D:\Downloads\vcc_data\vcc_data"
data_path = os.path.join(vcc_data_dir, 'adata_Training.h5ad')

if not os.path.exists(data_path):
    print(f"❌ Training data not found: {data_path}")
    sys.exit(1)

print("\n📊 Loading training data...")
adata = sc.read_h5ad(data_path)
print(f"✅ Training data loaded: {adata.shape}")

# =============================================================================
# STEP 2: IDENTIFY PERTURBATIONS TO VALIDATE
# =============================================================================
print("\n🎯 IDENTIFYING VALIDATION PERTURBATIONS")
print("-" * 40)

# Get perturbation counts
pert_counts = adata.obs['target_gene'].value_counts()
print(f"📊 Total perturbations in data: {len(pert_counts)}")

# Filter perturbations based on criteria
valid_perts = []
for pert_name, count in pert_counts.items():
    # Check if we have enough cells
    if count < MIN_CELLS_PER_PERT:
        continue
    
    # Check if we have ESM2 embedding
    if pert_name not in esm2_embeddings:
        continue
    
    # Skip non-targeting for now (handle separately)
    if pert_name == 'non-targeting' and not INCLUDE_CONTROLS:
        continue
    
    valid_perts.append(pert_name)

print(f"✅ Valid perturbations for validation: {len(valid_perts)}")

# Show some statistics
print(f"\n📈 Perturbation Statistics:")
for pert_name in valid_perts[:10]:  # Show first 10
    count = pert_counts[pert_name]
    print(f"   {pert_name}: {count} cells")

if len(valid_perts) > 10:
    print(f"   ... and {len(valid_perts) - 10} more")

# =============================================================================
# STEP 3: GENERATE PREDICTIONS FOR VALIDATION PERTURBATIONS
# =============================================================================
print("\n🔮 GENERATING VALIDATION PREDICTIONS")
print("-" * 40)

validation_results = {}

for i, pert_name in enumerate(valid_perts):
    print(f"\n🔄 Processing {pert_name} ({i+1}/{len(valid_perts)})")
    
    # Get cells for this perturbation
    pert_mask = adata.obs['target_gene'] == pert_name
    pert_cells = adata[pert_mask].copy()
    n_cells = pert_cells.shape[0]
    
    print(f"   📊 Found {n_cells} cells")
    
    # Limit number of cells for memory efficiency
    if n_cells > MAX_CELLS_PER_PERT:
        # Randomly sample cells
        sample_indices = np.random.choice(n_cells, MAX_CELLS_PER_PERT, replace=False)
        pert_cells = pert_cells[sample_indices]
        n_cells = MAX_CELLS_PER_PERT
        print(f"   🎲 Sampled {n_cells} cells")
    
    # Get real expression data
    if hasattr(pert_cells.X, 'toarray'):
        real_expression = pert_cells.X.toarray().astype(np.float32)
    else:
        real_expression = pert_cells.X.astype(np.float32)
    
    real_expression_log = np.log1p(real_expression)
    
    # Generate predictions using the model
    if pert_name in esm2_embeddings:
        pert_embedding = esm2_embeddings[pert_name].reshape(1, -1)
        
        # Predict differential expression profile
        pred_diff_profile = rf_model.predict(pert_embedding)[0]
        
        # Convert to absolute expression by adding control baseline
        pred_absolute_profile = pred_diff_profile + control_mean
        
        # Create predictions for all cells (same profile + noise)
        pred_expressions = np.tile(pred_absolute_profile, (n_cells, 1))
        
        # Add realistic noise to create cell-to-cell variation
        noise_std = 0.1  # Standard deviation of noise
        noise = np.random.normal(0, noise_std, pred_expressions.shape).astype(np.float32)
        pred_expressions += noise
        pred_expressions = np.maximum(pred_expressions, 0)  # Ensure non-negative
        
        print(f"   ✅ Generated predictions with noise (std={noise_std})")
    else:
        print(f"   ❌ No ESM2 embedding found for {pert_name}")
        continue
    
    # Store results
    validation_results[pert_name] = {
        'real_expression': real_expression_log,
        'pred_expression': pred_expressions,
        'cell_ids': pert_cells.obs.index.values,
        'n_cells': n_cells
    }
    
    print(f"   ✅ Stored validation data")
    
    # Memory cleanup
    del pert_cells, real_expression, real_expression_log, pred_expressions
    gc.collect()

print(f"\n✅ Generated predictions for {len(validation_results)} perturbations")

# =============================================================================
# STEP 4: SAVE VALIDATION DATA
# =============================================================================
print("\n💾 SAVING VALIDATION DATA")
print("-" * 25)

# Save individual perturbation files for detailed analysis
for pert_name, data in validation_results.items():
    pert_output_dir = os.path.join(OUTPUT_DIR, f"pert_{pert_name.replace('/', '_')}")
    os.makedirs(pert_output_dir, exist_ok=True)
    
    # Save real and predicted expression
    np.save(os.path.join(pert_output_dir, "real_expression.npy"), data['real_expression'])
    np.save(os.path.join(pert_output_dir, "pred_expression.npy"), data['pred_expression'])
    
    # Save metadata
    metadata = {
        'perturbation': pert_name,
        'n_cells': data['n_cells'],
        'cell_ids': data['cell_ids'].tolist()
    }
    
    pd.DataFrame(metadata).to_csv(os.path.join(pert_output_dir, "metadata.csv"), index=False)

print(f"✅ Saved individual perturbation files")

# Create combined validation dataset using memory-efficient approach
print("🔗 Creating combined validation dataset (memory-efficient)...")

# Calculate total dimensions
total_cells = sum(data['n_cells'] for data in validation_results.values())
n_genes = len(adata.var.index)

print(f"📊 Total dimensions: {total_cells:,} cells × {n_genes:,} genes")
print(f"📊 Estimated memory needed: {(total_cells * n_genes * 4) / 1e9:.2f} GB")

# Create metadata first
all_pert_names = []
all_cell_ids = []
for pert_name, data in validation_results.items():
    n_cells = data['n_cells']
    all_pert_names.extend([pert_name] * n_cells)
    all_cell_ids.extend([f"{pert_name}_{i}" for i in range(n_cells)])

obs_df = pd.DataFrame({
    'target_gene': all_pert_names,
    'cell_id': all_cell_ids,
    'dataset': 'validation'
})
obs_df.index = all_cell_ids

var_df = pd.DataFrame({
    'gene_id': adata.var.index,
    'feature_type': 'Gene Expression'
})
var_df.index = adata.var.index

print(f"✅ Created metadata for {len(all_cell_ids):,} cells")

# Save data in chunks to avoid memory issues
chunk_size = 5000  # Process 5000 cells at a time
n_chunks = (total_cells + chunk_size - 1) // chunk_size

print(f"🔄 Processing in {n_chunks} chunks of {chunk_size} cells...")

# Initialize temporary files for chunked storage
real_chunks = []
pred_chunks = []

current_cell_idx = 0
for chunk_idx in range(n_chunks):
    print(f"   Processing chunk {chunk_idx + 1}/{n_chunks}...")

    start_idx = current_cell_idx
    end_idx = min(current_cell_idx + chunk_size, total_cells)
    chunk_cells = end_idx - start_idx

    # Allocate chunk arrays
    real_chunk = np.zeros((chunk_cells, n_genes), dtype=np.float32)
    pred_chunk = np.zeros((chunk_cells, n_genes), dtype=np.float32)

    # Fill chunk with data from validation_results
    chunk_row = 0
    for pert_name, data in validation_results.items():
        pert_cells = data['n_cells']

        # Check if this perturbation overlaps with current chunk
        pert_start = sum(validation_results[p]['n_cells'] for p in list(validation_results.keys())[:list(validation_results.keys()).index(pert_name)])
        pert_end = pert_start + pert_cells

        # Calculate overlap
        overlap_start = max(start_idx, pert_start)
        overlap_end = min(end_idx, pert_end)

        if overlap_start < overlap_end:
            # There's overlap, copy the relevant data
            local_start = overlap_start - pert_start
            local_end = overlap_end - pert_start
            chunk_start = overlap_start - start_idx
            chunk_end = overlap_end - start_idx

            real_chunk[chunk_start:chunk_end] = data['real_expression'][local_start:local_end].astype(np.float32)
            pred_chunk[chunk_start:chunk_end] = data['pred_expression'][local_start:local_end].astype(np.float32)

    # Save chunks to temporary files
    real_chunk_path = os.path.join(OUTPUT_DIR, f"temp_real_chunk_{chunk_idx}.npy")
    pred_chunk_path = os.path.join(OUTPUT_DIR, f"temp_pred_chunk_{chunk_idx}.npy")

    np.save(real_chunk_path, real_chunk)
    np.save(pred_chunk_path, pred_chunk)

    real_chunks.append(real_chunk_path)
    pred_chunks.append(pred_chunk_path)

    current_cell_idx = end_idx

    # Clean up chunk arrays
    del real_chunk, pred_chunk
    gc.collect()

print("✅ All chunks processed and saved")

# Create AnnData objects by loading chunks incrementally
print("📦 Creating AnnData objects from chunks...")

# For AnnData, we'll use a different approach - create them with memory mapping
real_path = os.path.join(OUTPUT_DIR, "validation_real_data.h5ad")
pred_path = os.path.join(OUTPUT_DIR, "validation_predictions.h5ad")

# Method 1: Create smaller combined datasets by loading chunks and immediately saving
try:
    print("   📊 Combining real data chunks...")
    combined_real_chunks = []
    for chunk_path in real_chunks:
        chunk_data = np.load(chunk_path)
        combined_real_chunks.append(chunk_data)

    combined_real = np.vstack(combined_real_chunks)
    del combined_real_chunks
    gc.collect()

    real_adata = ad.AnnData(X=combined_real, obs=obs_df.copy(), var=var_df.copy())
    real_adata.write_h5ad(real_path)
    del combined_real, real_adata
    gc.collect()

    print("   📊 Combining predicted data chunks...")
    combined_pred_chunks = []
    for chunk_path in pred_chunks:
        chunk_data = np.load(chunk_path)
        combined_pred_chunks.append(chunk_data)

    combined_pred = np.vstack(combined_pred_chunks)
    del combined_pred_chunks
    gc.collect()

    pred_adata = ad.AnnData(X=combined_pred, obs=obs_df.copy(), var=var_df.copy())
    pred_adata.write_h5ad(pred_path)
    del combined_pred, pred_adata
    gc.collect()

    print("✅ Successfully created combined AnnData files")

except Exception as e:
    print(f"⚠️ Memory error creating combined files: {e}")
    print("💡 Individual perturbation files are still available for analysis")
    real_path = None
    pred_path = None

# Clean up temporary chunk files
print("🧹 Cleaning up temporary files...")
for chunk_path in real_chunks + pred_chunks:
    try:
        os.remove(chunk_path)
    except:
        pass

if real_path and pred_path:
    print(f"✅ Saved combined real data: {real_path}")
    print(f"✅ Saved combined predictions: {pred_path}")
else:
    print("⚠️ Combined files not created due to memory constraints")
    print("💡 Use individual perturbation files for analysis")

# Save summary statistics
summary_stats = []
for pert_name, data in validation_results.items():
    real_expr = data['real_expression']
    pred_expr = data['pred_expression']

    # Calculate basic statistics
    real_mean = np.mean(real_expr, axis=0)
    pred_mean = np.mean(pred_expr, axis=0)

    # Overall correlation
    correlation = np.corrcoef(real_mean, pred_mean)[0, 1]

    # Mean squared error
    mse = np.mean((real_mean - pred_mean) ** 2)

    summary_stats.append({
        'perturbation': pert_name,
        'n_cells': data['n_cells'],
        'correlation': correlation,
        'mse': mse,
        'real_mean_expr': np.mean(real_mean),
        'pred_mean_expr': np.mean(pred_mean)
    })

summary_df = pd.DataFrame(summary_stats)
summary_path = os.path.join(OUTPUT_DIR, "validation_summary.csv")
summary_df.to_csv(summary_path, index=False)

print(f"✅ Saved summary statistics: {summary_path}")

# =============================================================================
# STEP 5: GENERATE METADATA FOR VISUALIZATION
# =============================================================================
print("\n📋 GENERATING VISUALIZATION METADATA")
print("-" * 35)

# Create a metadata file for the visualization script
viz_metadata = {
    'model_type': 'ESM2_RandomForest',
    'model_path': rf_model_path,
    'control_mean_path': control_mean_path,
    'embeddings_path': esm2_embeddings_path,
    'n_perturbations': len(validation_results),
    'total_cells': len(all_cell_ids),
    'real_data_path': real_path if real_path else "Use individual perturbation files",
    'pred_data_path': pred_path if pred_path else "Use individual perturbation files",
    'summary_path': summary_path,
    'output_dir': OUTPUT_DIR,
    'gene_names': adata.var.index.tolist(),
    'perturbations': list(validation_results.keys()),
    'individual_files_available': True,
    'combined_files_available': bool(real_path and pred_path)
}

metadata_path = os.path.join(OUTPUT_DIR, "visualization_metadata.json")
import json
with open(metadata_path, 'w') as f:
    json.dump(viz_metadata, f, indent=2)

print(f"✅ Saved visualization metadata: {metadata_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n🎉 VALIDATION PREDICTIONS COMPLETE!")
print("=" * 40)

print(f"📊 SUMMARY:")
print(f"   🎯 Perturbations validated: {len(validation_results)}")
print(f"   🧬 Total cells: {len(all_cell_ids):,}")
print(f"   📁 Output directory: {OUTPUT_DIR}")

print(f"\n📁 GENERATED FILES:")
print(f"   📊 Real data: validation_real_data.h5ad")
print(f"   🔮 Predictions: validation_predictions.h5ad")
print(f"   📈 Summary stats: validation_summary.csv")
print(f"   📋 Metadata: visualization_metadata.json")
print(f"   📂 Individual perturbation folders: {len(validation_results)} folders")

print(f"\n📈 QUICK STATS:")
avg_correlation = summary_df['correlation'].mean()
avg_mse = summary_df['mse'].mean()
print(f"   📊 Average correlation: {avg_correlation:.4f}")
print(f"   📊 Average MSE: {avg_mse:.6f}")

best_pert = summary_df.loc[summary_df['correlation'].idxmax(), 'perturbation']
best_corr = summary_df['correlation'].max()
print(f"   🏆 Best prediction: {best_pert} (r={best_corr:.4f})")

print(f"\n🚀 NEXT STEPS:")
print(f"   1. Use visualization_metadata.json to create comparison plots")
print(f"   2. Analyze individual perturbations in their folders")
print(f"   3. Compare real vs predicted expression patterns")
print(f"   4. Identify which perturbations are predicted well/poorly")

print(f"\n💡 VISUALIZATION IDEAS:")
print(f"   📊 Correlation plots (real vs predicted)")
print(f"   🌋 Volcano plots (differential expression)")
print(f"   🗺️ UMAP/t-SNE embeddings comparison")
print(f"   📈 Gene expression heatmaps")
print(f"   📉 Perturbation-specific scatter plots")

# Cleanup
del validation_results, combined_real, combined_pred
gc.collect()
print(f"\n🧹 Memory cleaned up")