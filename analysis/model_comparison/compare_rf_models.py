"""
Compare predictions between ESM2 Random Forest models:
1. Absolute Expression Model (original)
2. Differential Expression Model (new approach)
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("🔬 COMPARING ESM2 RANDOM FOREST MODELS")
print("=" * 50)
print("📊 Absolute Expression vs Differential Expression Approaches")

# Configuration
FIGURES_DIR = "./outputs/figures/model_comparison"
os.makedirs(FIGURES_DIR, exist_ok=True)

# File paths based on your directory structure
absolute_submission = Path("vcc_esm2_rf_submission/esm2_rf_submission.h5ad")
differential_submission = Path("vcc_esm2_rf_differential_submission/esm2_rf_differential_submission.h5ad")

# Model info paths
absolute_model_info = Path("saved_esm2_rf_models/esm2_model_info.pkl")
differential_model_info = Path("saved_esm2_rf_differential_models/esm2_differential_model_info.pkl")

# =============================================================================
# STEP 1: LOAD AND VERIFY DATA
# =============================================================================
print("\n📁 LOADING MODEL SUBMISSIONS")
print("-" * 30)

# Check if files exist
if not absolute_submission.exists():
    print(f"❌ Absolute model submission not found: {absolute_submission}")
    exit(1)

if not differential_submission.exists():
    print(f"❌ Differential model submission not found: {differential_submission}")
    exit(1)

print("📊 Loading absolute expression model submission...")
adata_abs = ad.read_h5ad(absolute_submission)
print(f"   Shape: {adata_abs.shape}")

print("📊 Loading differential expression model submission...")
adata_diff = ad.read_h5ad(differential_submission)
print(f"   Shape: {adata_diff.shape}")

# Verify they have the same structure
print(f"\n🔍 VERIFYING DATA COMPATIBILITY:")
print(f"   Same number of genes: {adata_abs.shape[1] == adata_diff.shape[1]}")
print(f"   Same gene order: {np.array_equal(adata_abs.var.index, adata_diff.var.index)}")

# Get overlapping perturbations (they might have different cell counts)
abs_perts = set(adata_abs.obs['target_gene'].unique())
diff_perts = set(adata_diff.obs['target_gene'].unique())
common_perts = abs_perts.intersection(diff_perts)

print(f"   Absolute model perturbations: {len(abs_perts)}")
print(f"   Differential model perturbations: {len(diff_perts)}")
print(f"   Common perturbations: {len(common_perts)}")

# =============================================================================
# STEP 2: LOAD MODEL TRAINING INFO
# =============================================================================
print("\n📋 LOADING MODEL TRAINING INFO")
print("-" * 30)

import pickle

try:
    if absolute_model_info.exists():
        with open(absolute_model_info, 'rb') as f:
            abs_info = pickle.load(f)
        print("✅ Absolute model info loaded")
        print(f"   Test Pearson R: {abs_info.get('test_pearson', 'N/A'):.4f}")
        print(f"   Test MSE: {abs_info.get('test_mse', 'N/A'):.6f}")
    else:
        abs_info = {}
        print("⚠️ Absolute model info not found")

    if differential_model_info.exists():
        with open(differential_model_info, 'rb') as f:
            diff_info = pickle.load(f)
        print("✅ Differential model info loaded")
        print(f"   Test Pearson R: {diff_info.get('test_pearson', 'N/A'):.4f}")
        print(f"   Test MSE: {diff_info.get('test_mse', 'N/A'):.6f}")
        print(f"   Absolute Reconstruction R: {diff_info.get('absolute_reconstruction_pearson', 'N/A'):.4f}")
    else:
        diff_info = {}
        print("⚠️ Differential model info not found")

except Exception as e:
    print(f"⚠️ Error loading model info: {e}")
    abs_info = {}
    diff_info = {}

# =============================================================================
# STEP 3: COMPARE PERTURBATION-LEVEL PREDICTIONS
# =============================================================================
print("\n🧪 COMPARING PERTURBATION-LEVEL PREDICTIONS")
print("-" * 45)

# Calculate mean expression per perturbation for each model
perturbation_comparisons = []

print(f"📊 Analyzing {len(common_perts)} common perturbations...")

for i, pert in enumerate(sorted(common_perts)):
    if i % 10 == 0:
        print(f"   Processing perturbation {i+1}/{len(common_perts)}: {pert}")
    
    # Get cells for this perturbation from both models
    abs_mask = adata_abs.obs['target_gene'] == pert
    diff_mask = adata_diff.obs['target_gene'] == pert
    
    abs_cells = adata_abs[abs_mask]
    diff_cells = adata_diff[diff_mask]
    
    # Calculate mean expression profiles
    if hasattr(abs_cells.X, 'toarray'):
        abs_mean = np.mean(abs_cells.X.toarray(), axis=0)
        diff_mean = np.mean(diff_cells.X.toarray(), axis=0)
    else:
        abs_mean = np.mean(abs_cells.X, axis=0)
        diff_mean = np.mean(diff_cells.X, axis=0)
    
    # Calculate correlation between the two approaches
    correlation = pearsonr(abs_mean, diff_mean)[0]
    spearman_corr = spearmanr(abs_mean, diff_mean)[0]
    mse = np.mean((abs_mean - diff_mean) ** 2)
    mae = np.mean(np.abs(abs_mean - diff_mean))
    
    perturbation_comparisons.append({
        'perturbation': pert,
        'abs_cells': len(abs_cells),
        'diff_cells': len(diff_cells),
        'pearson_correlation': correlation,
        'spearman_correlation': spearman_corr,
        'mse': mse,
        'mae': mae,
        'abs_mean_expr': np.mean(abs_mean),
        'diff_mean_expr': np.mean(diff_mean)
    })

comparison_df = pd.DataFrame(perturbation_comparisons)
print(f"✅ Completed perturbation-level comparison")

# =============================================================================
# STEP 4: SUMMARY STATISTICS
# =============================================================================
print("\n📈 COMPARISON SUMMARY STATISTICS")
print("-" * 35)

print(f"🎯 OVERALL SIMILARITY:")
mean_correlation = comparison_df['pearson_correlation'].mean()
median_correlation = comparison_df['pearson_correlation'].median()
min_correlation = comparison_df['pearson_correlation'].min()
max_correlation = comparison_df['pearson_correlation'].max()

print(f"   Mean Pearson correlation: {mean_correlation:.4f}")
print(f"   Median Pearson correlation: {median_correlation:.4f}")
print(f"   Range: {min_correlation:.4f} to {max_correlation:.4f}")

mean_spearman = comparison_df['spearman_correlation'].mean()
print(f"   Mean Spearman correlation: {mean_spearman:.4f}")

print(f"\n📊 PREDICTION DIFFERENCES:")
mean_mse = comparison_df['mse'].mean()
mean_mae = comparison_df['mae'].mean()
print(f"   Mean MSE between models: {mean_mse:.6f}")
print(f"   Mean MAE between models: {mean_mae:.6f}")

# Identify most and least similar perturbations
most_similar = comparison_df.loc[comparison_df['pearson_correlation'].idxmax()]
least_similar = comparison_df.loc[comparison_df['pearson_correlation'].idxmin()]

print(f"\n🏆 MOST SIMILAR PERTURBATION:")
print(f"   {most_similar['perturbation']}: r={most_similar['pearson_correlation']:.4f}")

print(f"\n🤔 LEAST SIMILAR PERTURBATION:")
print(f"   {least_similar['perturbation']}: r={least_similar['pearson_correlation']:.4f}")

# =============================================================================
# STEP 5: VISUALIZATIONS
# =============================================================================
print("\n📊 CREATING COMPARISON VISUALIZATIONS")
print("-" * 35)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Plot 1: Distribution of correlations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ESM2 Random Forest Model Comparison\nAbsolute vs Differential Expression', fontsize=16, fontweight='bold')

# Correlation distribution
axes[0, 0].hist(comparison_df['pearson_correlation'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].axvline(mean_correlation, color='red', linestyle='--', label=f'Mean: {mean_correlation:.4f}')
axes[0, 0].axvline(median_correlation, color='orange', linestyle='--', label=f'Median: {median_correlation:.4f}')
axes[0, 0].set_xlabel('Pearson Correlation')
axes[0, 0].set_ylabel('Number of Perturbations')
axes[0, 0].set_title('Distribution of Per-Perturbation Correlations')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MSE distribution
axes[0, 1].hist(comparison_df['mse'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
axes[0, 1].axvline(mean_mse, color='red', linestyle='--', label=f'Mean: {mean_mse:.6f}')
axes[0, 1].set_xlabel('Mean Squared Error')
axes[0, 1].set_ylabel('Number of Perturbations')
axes[0, 1].set_title('Distribution of Prediction Differences (MSE)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Correlation vs MSE
axes[1, 0].scatter(comparison_df['pearson_correlation'], comparison_df['mse'], alpha=0.6, color='green')
axes[1, 0].set_xlabel('Pearson Correlation')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].set_title('Correlation vs Prediction Differences')
axes[1, 0].grid(True, alpha=0.3)

# Mean expression comparison
axes[1, 1].scatter(comparison_df['abs_mean_expr'], comparison_df['diff_mean_expr'], alpha=0.6, color='purple')
# Add diagonal line
min_expr = min(comparison_df['abs_mean_expr'].min(), comparison_df['diff_mean_expr'].min())
max_expr = max(comparison_df['abs_mean_expr'].max(), comparison_df['diff_mean_expr'].max())
axes[1, 1].plot([min_expr, max_expr], [min_expr, max_expr], 'r--', alpha=0.8)

expr_corr = pearsonr(comparison_df['abs_mean_expr'], comparison_df['diff_mean_expr'])[0]
axes[1, 1].set_xlabel('Absolute Model Mean Expression')
axes[1, 1].set_ylabel('Differential Model Mean Expression')
axes[1, 1].set_title(f'Mean Expression Levels\nr = {expr_corr:.4f}')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'model_comparison_overview.png'), dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Per-perturbation comparison
fig, ax = plt.subplots(figsize=(12, 8))

# Sort by correlation for better visualization
sorted_df = comparison_df.sort_values('pearson_correlation')
y_pos = np.arange(len(sorted_df))

# Color-code by correlation level
colors = ['red' if r < 0.8 else 'orange' if r < 0.9 else 'lightgreen' if r < 0.95 else 'darkgreen' 
          for r in sorted_df['pearson_correlation']]

bars = ax.barh(y_pos, sorted_df['pearson_correlation'], color=colors, alpha=0.7)

# Add perturbation names (show every 5th to avoid crowding)
step = max(1, len(sorted_df) // 20)  # Show at most 20 labels
ax.set_yticks(y_pos[::step])
ax.set_yticklabels(sorted_df['perturbation'].iloc[::step], fontsize=8)

ax.set_xlabel('Pearson Correlation')
ax.set_ylabel('Perturbations (sorted by correlation)')
ax.set_title('Per-Perturbation Correlation: Absolute vs Differential Models')
ax.grid(True, alpha=0.3)

# Add legend
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='r < 0.8 (Low)'),
    plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.7, label='0.8 ≤ r < 0.9 (Medium)'),
    plt.Rectangle((0,0),1,1, facecolor='lightgreen', alpha=0.7, label='0.9 ≤ r < 0.95 (High)'),
    plt.Rectangle((0,0),1,1, facecolor='darkgreen', alpha=0.7, label='r ≥ 0.95 (Very High)')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'per_perturbation_correlation.png'), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 6: DETAILED ANALYSIS OF EXTREME CASES
# =============================================================================
print("\n🔍 DETAILED ANALYSIS OF EXTREME CASES")
print("-" * 40)

# Analyze the most and least similar perturbations in detail
extreme_cases = [
    ('Most Similar', most_similar['perturbation']),
    ('Least Similar', least_similar['perturbation'])
]

for case_name, pert_name in extreme_cases:
    print(f"\n🔬 {case_name}: {pert_name}")
    
    # Get data for this perturbation
    abs_mask = adata_abs.obs['target_gene'] == pert_name
    diff_mask = adata_diff.obs['target_gene'] == pert_name
    
    abs_cells = adata_abs[abs_mask]
    diff_cells = adata_diff[diff_mask]
    
    # Get expression data
    if hasattr(abs_cells.X, 'toarray'):
        abs_expr = abs_cells.X.toarray()
        diff_expr = diff_cells.X.toarray()
    else:
        abs_expr = abs_cells.X
        diff_expr = diff_cells.X
    
    abs_mean = np.mean(abs_expr, axis=0)
    diff_mean = np.mean(diff_expr, axis=0)
    
    # Calculate statistics
    correlation = pearsonr(abs_mean, diff_mean)[0]
    mse = np.mean((abs_mean - diff_mean) ** 2)
    
    print(f"   Correlation: {correlation:.4f}")
    print(f"   MSE: {mse:.6f}")
    print(f"   Cells (Abs/Diff): {len(abs_cells)}/{len(diff_cells)}")
    print(f"   Expression range (Abs): {abs_mean.min():.3f} to {abs_mean.max():.3f}")
    print(f"   Expression range (Diff): {diff_mean.min():.3f} to {diff_mean.max():.3f}")
    
    # Create detailed comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{case_name} Perturbation: {pert_name}', fontsize=14, fontweight='bold')
    
    # Scatter plot
    axes[0].scatter(abs_mean, diff_mean, alpha=0.5, s=1)
    min_val = min(abs_mean.min(), diff_mean.min())
    max_val = max(abs_mean.max(), diff_mean.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0].set_xlabel('Absolute Model')
    axes[0].set_ylabel('Differential Model')
    axes[0].set_title(f'Expression Correlation\nr = {correlation:.4f}')
    axes[0].grid(True, alpha=0.3)
    
    # Top differential genes
    diff_values = abs_mean - diff_mean
    top_diff_genes = np.argsort(np.abs(diff_values))[-20:]
    
    x_pos = np.arange(len(top_diff_genes))
    axes[1].bar(x_pos, diff_values[top_diff_genes], alpha=0.7)
    axes[1].set_xlabel('Top Different Genes (rank)')
    axes[1].set_ylabel('Difference (Abs - Diff)')
    axes[1].set_title('Genes with Largest Differences')
    axes[1].grid(True, alpha=0.3)
    
    # Expression distributions
    axes[2].hist(abs_mean, bins=50, alpha=0.5, label='Absolute', density=True)
    axes[2].hist(diff_mean, bins=50, alpha=0.5, label='Differential', density=True)
    axes[2].set_xlabel('Expression Level')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Expression Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_name = pert_name.replace('/', '_').replace('\\', '_')
    filename = f'extreme_case_{case_name.lower().replace(" ", "_")}_{safe_name}.png'
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# STEP 7: SAVE COMPARISON RESULTS
# =============================================================================
print("\n💾 SAVING COMPARISON RESULTS")
print("-" * 30)

# Save comparison dataframe
comparison_path = os.path.join(FIGURES_DIR, 'model_comparison_results.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"✅ Saved comparison results: {comparison_path}")

# Create summary report
report = f"""
ESM2 RANDOM FOREST MODEL COMPARISON REPORT
{'='*60}

MODELS COMPARED:
• Absolute Expression Model: {absolute_submission}
• Differential Expression Model: {differential_submission}

DATASET COMPARISON:
• Common perturbations analyzed: {len(common_perts)}
• Total genes compared: {adata_abs.shape[1]:,}

SIMILARITY ANALYSIS:
• Mean Pearson correlation: {mean_correlation:.4f} ± {comparison_df['pearson_correlation'].std():.4f}
• Median Pearson correlation: {median_correlation:.4f}
• Correlation range: {min_correlation:.4f} to {max_correlation:.4f}
• Mean Spearman correlation: {mean_spearman:.4f}

PREDICTION DIFFERENCES:
• Mean MSE between models: {mean_mse:.6f}
• Mean MAE between models: {mean_mae:.6f}

EXTREME CASES:
• Most similar perturbation: {most_similar['perturbation']} (r={most_similar['pearson_correlation']:.4f})
• Least similar perturbation: {least_similar['perturbation']} (r={least_similar['pearson_correlation']:.4f})

TRAINING PERFORMANCE:
• Absolute model test R: {abs_info.get('test_pearson', 'N/A')}
• Differential model test R: {diff_info.get('test_pearson', 'N/A')}
• Differential model reconstruction R: {diff_info.get('absolute_reconstruction_pearson', 'N/A')}

INTERPRETATION:
The models show {'very high' if mean_correlation > 0.95 else 'high' if mean_correlation > 0.9 else 'moderate' if mean_correlation > 0.8 else 'low'} 
similarity (r = {mean_correlation:.4f}), suggesting that {'both approaches capture similar biological patterns' if mean_correlation > 0.9 else 'the approaches have meaningful differences'}.

GENERATED VISUALIZATIONS:
• model_comparison_overview.png - Overall comparison statistics
• per_perturbation_correlation.png - Per-perturbation correlation plot
• extreme_case_*.png - Detailed analysis of most/least similar cases
"""

report_path = os.path.join(FIGURES_DIR, 'model_comparison_report.txt')
with open(report_path, 'w') as f:
    f.write(report)

print(f"✅ Saved comparison report: {report_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n🎉 MODEL COMPARISON COMPLETE!")
print("=" * 40)

print(f"📊 SUMMARY:")
print(f"   🎯 Models compared: 2 (Absolute vs Differential)")
print(f"   🧪 Perturbations analyzed: {len(common_perts)}")
print(f"   📈 Mean correlation: {mean_correlation:.4f}")
print(f"   📁 Results saved to: {FIGURES_DIR}")

if mean_correlation > 0.95:
    print(f"\n✅ CONCLUSION: Very high similarity!")
    print(f"   💡 Both models produce nearly identical predictions")
    print(f"   🎯 Differential approach offers better interpretability without sacrificing accuracy")
elif mean_correlation > 0.9:
    print(f"\n✅ CONCLUSION: High similarity!")
    print(f"   💡 Models are very similar but with some interesting differences")
    print(f"   🎯 Differential approach may provide biological advantages")
else:
    print(f"\n🤔 CONCLUSION: Meaningful differences detected!")
    print(f"   💡 The two approaches produce different predictions")
    print(f"   🔬 Further investigation recommended")

print(f"\n📈 NEXT STEPS:")
print(f"   1. Examine detailed visualizations in {FIGURES_DIR}")
print(f"   2. Investigate perturbations with low correlation")
print(f"   3. Consider biological interpretation of differences")
print(f"   4. Test both models on validation data")
print(f"   5. Submit both approaches to VCC for comparison")