#!/usr/bin/env python3
"""
VCC Evaluation and Visualization System
- PCA visualization of predicted vs actual cells
- Estimate official VCC scores (DE, PDisc, MAE)
- Compare multiple models
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support
from scipy.stats import mannwhitneyu, rankdata
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

class VCCEvaluator:
    """VCC-style evaluation and visualization"""
    
    def __init__(self, true_data_path, predicted_data_path, control_name='non-targeting'):
        """
        Initialize evaluator
        
        Args:
            true_data_path: Path to ground truth training data
            predicted_data_path: Path to model predictions
            control_name: Name of control perturbation
        """
        self.control_name = control_name
        
        print("📊 INITIALIZING VCC EVALUATOR")
        print("=" * 40)
        
        # Load true data
        print("📥 Loading ground truth data...")
        self.true_data = sc.read_h5ad(true_data_path)
        print(f"   Shape: {self.true_data.shape}")
        
        # Load predicted data  
        print("📥 Loading predicted data...")
        self.pred_data = sc.read_h5ad(predicted_data_path)
        print(f"   Shape: {self.pred_data.shape}")
        
        # Extract perturbations present in both datasets
        true_perts = set(self.true_data.obs['target_gene'].unique())
        pred_perts = set(self.pred_data.obs['target_gene'].unique())
        self.common_perts = list(true_perts.intersection(pred_perts))
        
        print(f"✅ Common perturbations: {len(self.common_perts)}")
        print(f"   Sample: {self.common_perts[:5]}")
        
        # Store results
        self.scores = {}
        self.pca_results = {}
        
    def compute_differential_expression_score(self, alpha=0.05, max_cells_per_pert=500):
        """
        Compute Differential Expression Score following VCC methodology
        
        Args:
            alpha: FDR threshold for DE gene calling
            max_cells_per_pert: Sample cells to speed up computation
        """
        
        print(f"\n🧬 COMPUTING DIFFERENTIAL EXPRESSION SCORE")
        print("-" * 45)
        
        de_scores = []
        
        # Get control cells from true data
        true_control_mask = self.true_data.obs['target_gene'] == self.control_name
        true_control_expr = self.true_data.X[true_control_mask]
        if hasattr(true_control_expr, 'toarray'):
            true_control_expr = true_control_expr.toarray()
        
        # Get control cells from predicted data
        pred_control_mask = self.pred_data.obs['target_gene'] == self.control_name
        pred_control_expr = self.pred_data.X[pred_control_mask]
        if hasattr(pred_control_expr, 'toarray'):
            pred_control_expr = pred_control_expr.toarray()
        
        print(f"   Control cells - True: {true_control_expr.shape[0]}, Pred: {pred_control_expr.shape[0]}")
        
        for i, pert in enumerate(self.common_perts):
            if pert == self.control_name:
                continue
                
            print(f"   Processing {pert} ({i+1}/{len(self.common_perts)})...")
            
            # Get perturbation cells
            true_pert_mask = self.true_data.obs['target_gene'] == pert
            true_pert_expr = self.true_data.X[true_pert_mask]
            if hasattr(true_pert_expr, 'toarray'):
                true_pert_expr = true_pert_expr.toarray()
            
            pred_pert_mask = self.pred_data.obs['target_gene'] == pert
            pred_pert_expr = self.pred_data.X[pred_pert_mask]
            if hasattr(pred_pert_expr, 'toarray'):
                pred_pert_expr = pred_pert_expr.toarray()
            
            # Sample cells if too many
            if true_pert_expr.shape[0] > max_cells_per_pert:
                sample_idx = np.random.choice(true_pert_expr.shape[0], max_cells_per_pert, replace=False)
                true_pert_expr = true_pert_expr[sample_idx]
            
            if pred_pert_expr.shape[0] > max_cells_per_pert:
                sample_idx = np.random.choice(pred_pert_expr.shape[0], max_cells_per_pert, replace=False)
                pred_pert_expr = pred_pert_expr[sample_idx]
            
            # Wilcoxon test for each gene - TRUE DATA
            true_pvals = []
            for gene_idx in range(self.true_data.shape[1]):
                try:
                    _, pval = mannwhitneyu(
                        true_pert_expr[:, gene_idx], 
                        true_control_expr[:, gene_idx],
                        alternative='two-sided'
                    )
                    true_pvals.append(pval)
                except:
                    true_pvals.append(1.0)
            
            # Wilcoxon test for each gene - PREDICTED DATA  
            pred_pvals = []
            for gene_idx in range(self.pred_data.shape[1]):
                try:
                    _, pval = mannwhitneyu(
                        pred_pert_expr[:, gene_idx],
                        pred_control_expr[:, gene_idx], 
                        alternative='two-sided'
                    )
                    pred_pvals.append(pval)
                except:
                    pred_pvals.append(1.0)
            
            # FDR correction
            true_rejected, _, _, _ = multipletests(true_pvals, alpha=alpha, method='fdr_bh')
            pred_rejected, _, _, _ = multipletests(pred_pvals, alpha=alpha, method='fdr_bh')
            
            # Calculate overlap (F1 score)
            true_positive = np.sum(true_rejected & pred_rejected)
            false_positive = np.sum(~true_rejected & pred_rejected)  
            false_negative = np.sum(true_rejected & ~pred_rejected)
            
            if true_positive + false_positive > 0:
                precision = true_positive / (true_positive + false_positive)
            else:
                precision = 0.0
                
            if true_positive + false_negative > 0:
                recall = true_positive / (true_positive + false_negative)
            else:
                recall = 0.0
                
            if precision + recall > 0:
                f1_score = 2 * precision * recall / (precision + recall)
            else:
                f1_score = 0.0
            
            de_scores.append({
                'perturbation': pert,
                'n_true_de': np.sum(true_rejected),
                'n_pred_de': np.sum(pred_rejected),
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            })
        
        # Aggregate results
        avg_de_score = np.mean([s['f1_score'] for s in de_scores])
        self.scores['de_score'] = avg_de_score
        self.scores['de_details'] = de_scores
        
        print(f"✅ Average DE Score: {avg_de_score:.4f}")
        return avg_de_score
    
    def compute_perturbation_discrimination_score(self, max_cells_per_pert=200):
        """
        Compute Perturbation Discrimination Score
        """
        
        print(f"\n🎯 COMPUTING PERTURBATION DISCRIMINATION SCORE")
        print("-" * 50)
        
        # Create pseudo-bulk profiles for all perturbations
        true_profiles = {}
        pred_profiles = {}
        
        for pert in self.common_perts:
            # True data pseudo-bulk
            true_mask = self.true_data.obs['target_gene'] == pert
            true_cells = self.true_data.X[true_mask]
            if hasattr(true_cells, 'toarray'):
                true_cells = true_cells.toarray()
                
            # Sample if too many cells
            if true_cells.shape[0] > max_cells_per_pert:
                sample_idx = np.random.choice(true_cells.shape[0], max_cells_per_pert, replace=False)
                true_cells = true_cells[sample_idx]
            
            true_profiles[pert] = np.mean(true_cells, axis=0)
            
            # Predicted data pseudo-bulk
            pred_mask = self.pred_data.obs['target_gene'] == pert
            pred_cells = self.pred_data.X[pred_mask]
            if hasattr(pred_cells, 'toarray'):
                pred_cells = pred_cells.toarray()
                
            if pred_cells.shape[0] > max_cells_per_pert:
                sample_idx = np.random.choice(pred_cells.shape[0], max_cells_per_pert, replace=False)
                pred_cells = pred_cells[sample_idx]
                
            pred_profiles[pert] = np.mean(pred_cells, axis=0)
        
        # Compute discrimination for each perturbation
        discrimination_scores = []
        
        for target_pert in self.common_perts:
            if target_pert == self.control_name:
                continue
                
            pred_profile = pred_profiles[target_pert]
            
            # Compute Manhattan distances to all true profiles
            distances = {}
            for true_pert in self.common_perts:
                true_profile = true_profiles[true_pert]
                distance = np.sum(np.abs(pred_profile - true_profile))
                distances[true_pert] = distance
            
            # Rank distances (lower = better)
            sorted_distances = sorted(distances.items(), key=lambda x: x[1])
            
            # Find rank of true perturbation
            true_rank = next(i for i, (pert, _) in enumerate(sorted_distances) if pert == target_pert)
            
            # Normalize by number of perturbations
            discrimination_score = true_rank / (len(self.common_perts) - 1)
            discrimination_scores.append(discrimination_score)
        
        # Average and transform (VCC uses 1 - 2*score)
        avg_pdisc = np.mean(discrimination_scores)
        pdisc_norm = 1 - 2 * avg_pdisc
        
        self.scores['pdisc_raw'] = avg_pdisc
        self.scores['pdisc_norm'] = pdisc_norm
        
        print(f"✅ Raw PDisc Score: {avg_pdisc:.4f}")
        print(f"✅ Normalized PDisc Score: {pdisc_norm:.4f}")
        
        return pdisc_norm
    
    def compute_mae_score(self, max_cells_per_pert=200):
        """
        Compute Mean Absolute Error
        """
        
        print(f"\n📏 COMPUTING MEAN ABSOLUTE ERROR")
        print("-" * 35)
        
        all_true = []
        all_pred = []
        
        for pert in self.common_perts:
            # True data
            true_mask = self.true_data.obs['target_gene'] == pert
            true_cells = self.true_data.X[true_mask]
            if hasattr(true_cells, 'toarray'):
                true_cells = true_cells.toarray()
            
            # Predicted data  
            pred_mask = self.pred_data.obs['target_gene'] == pert
            pred_cells = self.pred_data.X[pred_mask]
            if hasattr(pred_cells, 'toarray'):
                pred_cells = pred_cells.toarray()
            
            # Sample to same size
            min_cells = min(true_cells.shape[0], pred_cells.shape[0], max_cells_per_pert)
            
            if true_cells.shape[0] > min_cells:
                true_idx = np.random.choice(true_cells.shape[0], min_cells, replace=False)
                true_cells = true_cells[true_idx]
            
            if pred_cells.shape[0] > min_cells:
                pred_idx = np.random.choice(pred_cells.shape[0], min_cells, replace=False)
                pred_cells = pred_cells[pred_idx]
            
            all_true.append(true_cells)
            all_pred.append(pred_cells)
        
        # Combine all cells
        all_true = np.vstack(all_true)
        all_pred = np.vstack(all_pred)
        
        # Compute MAE
        mae = mean_absolute_error(all_true.flatten(), all_pred.flatten())
        self.scores['mae'] = mae
        
        print(f"✅ MAE Score: {mae:.4f}")
        return mae
    
    def compute_overall_score(self, weights=None):
        """
        Compute overall VCC-style score
        """
        
        if weights is None:
            # Default weights (approximate - actual VCC weights are not public)
            weights = {'de': 0.4, 'pdisc': 0.4, 'mae': 0.2}
        
        # Normalize MAE (lower is better, so invert)
        # This is approximate - VCC uses baseline normalization
        mae_normalized = 1 / (1 + self.scores['mae'])
        
        overall = (weights['de'] * self.scores['de_score'] + 
                  weights['pdisc'] * self.scores['pdisc_norm'] +
                  weights['mae'] * mae_normalized)
        
        self.scores['overall'] = overall
        
        print(f"\n🏆 OVERALL VCC-STYLE SCORE: {overall:.4f}")
        print(f"   DE Score: {self.scores['de_score']:.4f} (weight: {weights['de']})")
        print(f"   PDisc Score: {self.scores['pdisc_norm']:.4f} (weight: {weights['pdisc']})")
        print(f"   MAE Score: {mae_normalized:.4f} (weight: {weights['mae']})")
        
        return overall
    
    def create_pca_visualization(self, n_components=2, max_cells_per_pert=100, selected_perts=None):
        """
        Create PCA visualization comparing predicted vs actual cells
        """
        
        print(f"\n📊 CREATING PCA VISUALIZATION")
        print("-" * 35)
        
        if selected_perts is None:
            # Select most interesting perturbations (exclude control, take top 8)
            selected_perts = [p for p in self.common_perts[:9] if p != self.control_name]
        
        print(f"   Visualizing: {selected_perts}")
        
        # Collect data for PCA
        pca_data = []
        pca_labels = []
        pca_types = []
        
        for pert in selected_perts + [self.control_name]:
            # True data
            true_mask = self.true_data.obs['target_gene'] == pert
            true_cells = self.true_data.X[true_mask]
            if hasattr(true_cells, 'toarray'):
                true_cells = true_cells.toarray()
            
            # Sample cells
            if true_cells.shape[0] > max_cells_per_pert:
                sample_idx = np.random.choice(true_cells.shape[0], max_cells_per_pert, replace=False)
                true_cells = true_cells[sample_idx]
            
            pca_data.append(true_cells)
            pca_labels.extend([pert] * true_cells.shape[0])
            pca_types.extend(['True'] * true_cells.shape[0])
            
            # Predicted data
            pred_mask = self.pred_data.obs['target_gene'] == pert
            pred_cells = self.pred_data.X[pred_mask]
            if hasattr(pred_cells, 'toarray'):
                pred_cells = pred_cells.toarray()
            
            if pred_cells.shape[0] > max_cells_per_pert:
                sample_idx = np.random.choice(pred_cells.shape[0], max_cells_per_pert, replace=False)
                pred_cells = pred_cells[sample_idx]
            
            pca_data.append(pred_cells)
            pca_labels.extend([pert] * pred_cells.shape[0])
            pca_types.extend(['Predicted'] * pred_cells.shape[0])
        
        # Combine data
        all_data = np.vstack(pca_data)
        
        # Apply PCA
        print(f"   Fitting PCA on {all_data.shape[0]} cells, {all_data.shape[1]} genes...")
        pca = PCA(n_components=n_components, random_state=42)
        pca_coords = pca.fit_transform(all_data)
        
        # Create DataFrame for plotting
        pca_df = pd.DataFrame({
            'PC1': pca_coords[:, 0],
            'PC2': pca_coords[:, 1] if n_components > 1 else np.zeros(len(pca_coords)),
            'Perturbation': pca_labels,
            'Type': pca_types
        })
        
        # Store results
        self.pca_results = {
            'data': pca_df,
            'pca_model': pca,
            'variance_explained': pca.explained_variance_ratio_
        }
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Colored by perturbation
        for pert in selected_perts + [self.control_name]:
            pert_data = pca_df[pca_df['Perturbation'] == pert]
            true_data = pert_data[pert_data['Type'] == 'True']
            pred_data = pert_data[pert_data['Type'] == 'Predicted']
            
            # Plot true cells as circles
            axes[0].scatter(true_data['PC1'], true_data['PC2'], 
                           label=f'{pert} (True)', alpha=0.6, s=20)
            
            # Plot predicted cells as triangles
            axes[0].scatter(pred_data['PC1'], pred_data['PC2'], 
                           label=f'{pert} (Pred)', alpha=0.8, s=30, marker='^')
        
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0].set_title('PCA: Predicted vs True Cells by Perturbation')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Colored by type (True vs Predicted)
        for cell_type in ['True', 'Predicted']:
            type_data = pca_df[pca_df['Type'] == cell_type]
            marker = 'o' if cell_type == 'True' else '^'
            axes[1].scatter(type_data['PC1'], type_data['PC2'], 
                           label=cell_type, alpha=0.6, s=20, marker=marker)
        
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[1].set_title('PCA: True vs Predicted Cells')
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save plot
        output_dir = "vcc_evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/pca_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ PCA visualization saved to {output_dir}/pca_visualization.png")
        
        return pca_df
    
    def generate_evaluation_report(self):
        """
        Generate comprehensive evaluation report
        """
        
        print(f"\n📋 VCC EVALUATION REPORT")
        print("=" * 50)
        
        # Summary scores
        print(f"🏆 SUMMARY SCORES:")
        print(f"   Overall Score: {self.scores.get('overall', 'TBD'):.4f}")
        print(f"   DE Score: {self.scores.get('de_score', 'TBD'):.4f}")
        print(f"   PDisc Score: {self.scores.get('pdisc_norm', 'TBD'):.4f}")
        print(f"   MAE Score: {self.scores.get('mae', 'TBD'):.4f}")
        
        # Detailed breakdown
        if 'de_details' in self.scores:
            print(f"\n🧬 DIFFERENTIAL EXPRESSION DETAILS:")
            de_df = pd.DataFrame(self.scores['de_details'])
            print(f"   Average precision: {de_df['precision'].mean():.3f}")
            print(f"   Average recall: {de_df['recall'].mean():.3f}")
            print(f"   Top performing perturbations:")
            top_de = de_df.nlargest(3, 'f1_score')
            for _, row in top_de.iterrows():
                print(f"     {row['perturbation']}: F1={row['f1_score']:.3f}")
        
        # Comparison to known benchmarks
        print(f"\n📊 PERFORMANCE ASSESSMENT:")
        
        overall_score = self.scores.get('overall', 0)
        if overall_score > 0.5:
            print(f"   🎉 Excellent performance! (Overall > 0.5)")
        elif overall_score > 0.2:
            print(f"   ✅ Good performance (Overall > 0.2)")
        elif overall_score > 0.1:
            print(f"   ⚠️ Moderate performance (Overall > 0.1)")
        else:
            print(f"   ❌ Needs improvement (Overall < 0.1)")
        
        # Save detailed results
        output_dir = "vcc_evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/evaluation_scores.json", 'w') as f:
            import json
            # Convert numpy types for JSON serialization
            scores_json = {}
            for key, value in self.scores.items():
                if isinstance(value, (np.ndarray, np.float64, np.float32)):
                    scores_json[key] = float(value)
                elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    scores_json[key] = value
                else:
                    scores_json[key] = value
            json.dump(scores_json, f, indent=2)
        
        print(f"\n💾 Results saved to {output_dir}/")
        return self.scores

def main():
    """
    Example usage of VCC Evaluator
    """
    
    print("🚀 VCC EVALUATION PIPELINE")
    print("=" * 40)
    
    # Define paths (update these to match your data)
    true_data_path = "data/raw/single_cell_rnaseq/vcc_data/adata_Training.h5ad"
    pred_data_path = "vcc_esm2_rf_submission/esm2_rf_submission.h5ad"
    
    try:
        # Initialize evaluator
        evaluator = VCCEvaluator(true_data_path, pred_data_path)
        
        # Run evaluation pipeline
        evaluator.compute_differential_expression_score()
        evaluator.compute_perturbation_discrimination_score()
        evaluator.compute_mae_score()
        evaluator.compute_overall_score()
        
        # Create visualizations
        evaluator.create_pca_visualization()
        
        # Generate report
        evaluator.generate_evaluation_report()
        
        print(f"\n🎉 EVALUATION COMPLETE!")
        print(f"✅ Check vcc_evaluation_results/ for detailed outputs")
        
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        print(f"💡 Make sure your data paths are correct:")
        print(f"   True data: {true_data_path}")
        print(f"   Predicted data: {pred_data_path}")

if __name__ == "__main__":
    main()