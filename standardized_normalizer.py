#!/usr/bin/env python3
"""
Median Non-Zero Normalizer for Standardized Datasets
Uses the gene-name-standardized datasets for proper cross-dataset analysis
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pickle
import warnings

warnings.filterwarnings('ignore')

class StandardizedDatasetNormalizer:
    """
    Normalize expression using standardized gene names across datasets
    """
    
    def __init__(self, 
                 standardized_dir="./standardized_datasets",
                 output_dir="./standardized_normalization_results",
                 min_expression_threshold: float = 0.1):
        """
        Initialize normalizer for standardized datasets
        """
        self.standardized_dir = Path(standardized_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_expression_threshold = min_expression_threshold
        
        # Store results
        self.control_profiles = {}
        self.normalization_stats = {}
        
        # Dataset control strategies
        self.dataset_strategies = {
            'VCC_Training_Subset': {'column': 'target_gene', 'identifiers': ['non-targeting']},
            '264667_hepg2': {'column': 'gene', 'identifiers': ['non-targeting']},
            '264667_jurkat': {'column': 'gene', 'identifiers': ['non-targeting']},
            'K562_essential': {'column': 'gene', 'identifiers': ['non-targeting']},
            '274751_tfko.sng.guides.full.ct': {'column': 'WT', 'control_value': 'T'}
        }
    
    def find_standardized_datasets(self):
        """
        Find all standardized datasets
        """
        print(f"📁 Looking for standardized datasets in: {self.standardized_dir}")
        
        if not self.standardized_dir.exists():
            print(f"❌ Standardized datasets directory not found!")
            print(f"💡 Run the gene converter first to create standardized datasets")
            return []
        
        # Find standardized files
        standardized_files = list(self.standardized_dir.glob("*_standardized.h5ad"))
        
        if not standardized_files:
            print(f"❌ No *_standardized.h5ad files found!")
            return []
        
        print(f"✅ Found {len(standardized_files)} standardized datasets:")
        dataset_info = []
        
        for file_path in standardized_files:
            dataset_name = file_path.stem.replace('_standardized', '')
            file_size = file_path.stat().st_size / (1024**3)  # GB
            
            print(f"   📊 {dataset_name}: {file_size:.1f} GB")
            dataset_info.append((str(file_path), dataset_name))
        
        return dataset_info
    
    def identify_control_cells(self, adata: ad.AnnData, dataset_name: str) -> np.ndarray:
        """
        Identify control cells using dataset-specific strategy
        """
        print(f"🔍 Identifying control cells for {dataset_name}...")
        
        strategy = self.dataset_strategies.get(dataset_name, {})
        
        if not strategy:
            # Try common strategies if dataset not in predefined list
            common_strategies = [
                {'column': 'target_gene', 'identifiers': ['non-targeting']},
                {'column': 'gene', 'identifiers': ['non-targeting']},
                {'column': 'WT', 'control_value': 'T'}
            ]
            
            for test_strategy in common_strategies:
                if 'identifiers' in test_strategy:
                    col = test_strategy['column']
                    if col in adata.obs.columns:
                        mask = adata.obs[col].astype(str).str.lower().isin(
                            [id.lower() for id in test_strategy['identifiers']]
                        )
                        if mask.sum() > 0:
                            print(f"   ✅ Found {mask.sum():,} controls using {col}")
                            return mask
                else:
                    col = test_strategy['column']
                    if col in adata.obs.columns:
                        mask = (adata.obs[col] == test_strategy['control_value'])
                        if mask.sum() > 0:
                            print(f"   ✅ Found {mask.sum():,} controls using {col}")
                            return mask
            
            print(f"   ❌ No control cells found for {dataset_name}")
            return np.zeros(adata.n_obs, dtype=bool)
        
        control_mask = np.zeros(adata.n_obs, dtype=bool)
        
        if 'identifiers' in strategy:
            col = strategy['column']
            if col in adata.obs.columns:
                col_data = adata.obs[col].astype(str).str.lower()
                for identifier in strategy['identifiers']:
                    mask = (col_data == identifier.lower())
                    control_mask |= mask
                    if mask.sum() > 0:
                        print(f"   ✅ Found {mask.sum():,} cells with '{identifier}' in {col}")
        
        elif 'control_value' in strategy:
            col = strategy['column']
            if col in adata.obs.columns:
                control_mask = (adata.obs[col] == strategy['control_value'])
                print(f"   ✅ Found {control_mask.sum():,} cells where {col}=={strategy['control_value']}")
        
        return control_mask
    
    def create_normalized_profile(self, adata: ad.AnnData, 
                                control_mask: np.ndarray, 
                                dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create pseudobulk control profile with median non-zero normalization
        """
        print(f"🧮 Creating normalized profile for {dataset_name}...")
        
        if control_mask.sum() == 0:
            return None, None
        
        # Extract control cells
        adata_control = adata[control_mask]
        
        # Get expression matrix
        if sparse.issparse(adata_control.X):
            X = adata_control.X.toarray().astype(np.float32)
        else:
            X = adata_control.X.astype(np.float32)
        
        # Create pseudobulk profile
        raw_profile = np.mean(X, axis=0)
        
        print(f"   📊 Pseudobulk from {control_mask.sum():,} control cells")
        
        # Apply median non-zero normalization
        nonzero_values = raw_profile[raw_profile > self.min_expression_threshold]
        
        if len(nonzero_values) > 0:
            median_nonzero = np.median(nonzero_values)
            normalized_profile = raw_profile / median_nonzero
            
            print(f"   📊 Median non-zero: {median_nonzero:.4f}")
            print(f"   📊 Non-zero genes: {len(nonzero_values):,}/{len(raw_profile):,}")
            
            # Store stats
            self.normalization_stats[dataset_name] = {
                'n_control_cells': control_mask.sum(),
                'n_genes': len(raw_profile),
                'n_nonzero_genes': len(nonzero_values),
                'median_nonzero_raw': median_nonzero,
                'mean_raw': np.mean(raw_profile),
                'mean_normalized': np.mean(normalized_profile),
                'max_normalized': np.max(normalized_profile),
                'std_normalized': np.std(normalized_profile)
            }
            
        else:
            normalized_profile = raw_profile.copy()
            median_nonzero = 1.0
            
            self.normalization_stats[dataset_name] = {
                'n_control_cells': control_mask.sum(),
                'n_genes': len(raw_profile),
                'n_nonzero_genes': 0,
                'median_nonzero_raw': median_nonzero,
                'mean_raw': np.mean(raw_profile),
                'mean_normalized': np.mean(normalized_profile),
                'max_normalized': np.max(normalized_profile),
                'std_normalized': np.std(normalized_profile)
            }
        
        return raw_profile, normalized_profile
    
    def process_dataset(self, dataset_path: str, dataset_name: str) -> bool:
        """
        Process a single standardized dataset
        """
        print(f"\n{'='*60}")
        print(f"🔄 Processing: {dataset_name}")
        print(f"📁 Dataset: {dataset_path}")
        print(f"{'='*60}")
        
        try:
            # Load standardized dataset
            adata = sc.read_h5ad(dataset_path)
            print(f"📊 Shape: {adata.shape}")
            print(f"🧬 Sample genes: {list(adata.var.index[:5])}")
            
            # Check gene format (should be mostly symbols now)
            sample_genes = list(adata.var.index[:10])
            ensembl_count = sum(1 for g in sample_genes if str(g).startswith('ENSG'))
            symbol_count = sum(1 for g in sample_genes if str(g).isalpha() and not str(g).startswith('ENSG'))
            
            print(f"📋 Gene format: {ensembl_count} Ensembl, {symbol_count} symbols (sample of 10)")
            
            # Identify control cells
            control_mask = self.identify_control_cells(adata, dataset_name)
            
            if control_mask.sum() == 0:
                print("❌ No control cells found, skipping")
                return False
            
            # Create normalized profiles
            raw_profile, normalized_profile = self.create_normalized_profile(
                adata, control_mask, dataset_name
            )
            
            if raw_profile is None:
                return False
            
            # Store results
            self.control_profiles[dataset_name] = {
                'raw_profile': raw_profile,
                'normalized_profile': normalized_profile,
                'gene_names': adata.var.index.values.copy()
            }
            
            # Save individual results
            dataset_dir = self.output_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            np.save(dataset_dir / "raw_control_profile.npy", raw_profile)
            np.save(dataset_dir / "normalized_control_profile.npy", normalized_profile)
            
            with open(dataset_dir / "gene_names.txt", 'w', encoding='utf-8') as f:
                for gene in adata.var.index:
                    f.write(f"{gene}\n")
            
            with open(dataset_dir / "normalization_stats.pkl", 'wb') as f:
                pickle.dump(self.normalization_stats[dataset_name], f)
            
            print(f"✅ Successfully processed {dataset_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            return False
    
    def analyze_gene_overlap(self):
        """
        Analyze gene overlap across standardized datasets
        """
        print(f"\n🔍 ANALYZING GENE OVERLAP (STANDARDIZED)")
        print("=" * 60)
        
        if len(self.control_profiles) < 2:
            print("⚠️ Need at least 2 datasets for overlap analysis")
            return
        
        # Find common genes
        all_gene_sets = []
        for dataset_name, profile_data in self.control_profiles.items():
            gene_set = set(profile_data['gene_names'])
            all_gene_sets.append(gene_set)
            print(f"📊 {dataset_name}: {len(gene_set):,} genes")
        
        # Calculate overlaps
        common_genes = set.intersection(*all_gene_sets)
        union_genes = set.union(*all_gene_sets)
        
        print(f"\n🎯 OVERLAP RESULTS:")
        print(f"   Common to ALL datasets: {len(common_genes):,} genes")
        print(f"   Total unique genes: {len(union_genes):,} genes")
        print(f"   Overlap percentage: {len(common_genes)/len(union_genes)*100:.1f}%")
        
        if len(common_genes) > 0:
            print(f"\n📋 Sample common genes:")
            common_sorted = sorted(list(common_genes))
            for i, gene in enumerate(common_sorted[:20], 1):
                print(f"   {i:2d}. {gene}")
            
            if len(common_genes) > 20:
                print(f"   ... and {len(common_genes)-20:,} more")
        
        return common_genes
    
    def create_visualizations(self, common_genes):
        """
        Create visualizations with standardized gene overlap
        """
        print(f"\n📊 Creating visualizations...")
        
        if len(common_genes) == 0:
            print("❌ No common genes for visualization")
            return
        
        # Select genes for plotting
        genes_to_plot = sorted(list(common_genes))[:100]
        print(f"   📊 Plotting {len(genes_to_plot)} common genes")
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Standardized Cross-Dataset Normalization Results', fontsize=16)
        
        dataset_names = list(self.control_profiles.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(dataset_names)))
        
        # Plot 1: Raw expression comparison
        ax = axes[0, 0]
        for i, dataset_name in enumerate(dataset_names):
            profile_data = self.control_profiles[dataset_name]
            gene_names = list(profile_data['gene_names'])
            raw_profile = profile_data['raw_profile']
            
            # Get values for common genes
            gene_values = []
            for gene in genes_to_plot:
                if gene in gene_names:
                    gene_idx = gene_names.index(gene)
                    gene_values.append(raw_profile[gene_idx])
                else:
                    gene_values.append(0)
            
            x_pos = np.arange(len(genes_to_plot)) + i * 0.15
            ax.bar(x_pos, gene_values, width=0.15, label=dataset_name, 
                   alpha=0.7, color=colors[i])
        
        ax.set_xlabel('Common Genes')
        ax.set_ylabel('Raw Expression (log scale)')
        ax.set_title('Raw Expression Profiles')
        ax.set_yscale('log')
        ax.legend()
        
        # Plot 2: Normalized expression comparison  
        ax = axes[0, 1]
        for i, dataset_name in enumerate(dataset_names):
            profile_data = self.control_profiles[dataset_name]
            gene_names = list(profile_data['gene_names'])
            normalized_profile = profile_data['normalized_profile']
            
            gene_values = []
            for gene in genes_to_plot:
                if gene in gene_names:
                    gene_idx = gene_names.index(gene)
                    gene_values.append(normalized_profile[gene_idx])
                else:
                    gene_values.append(0)
            
            x_pos = np.arange(len(genes_to_plot)) + i * 0.15
            ax.bar(x_pos, gene_values, width=0.15, label=dataset_name, 
                   alpha=0.7, color=colors[i])
        
        ax.set_xlabel('Common Genes')
        ax.set_ylabel('Normalized Expression')
        ax.set_title('Normalized Expression Profiles')
        ax.legend()
        
        # Plot 3: Correlation heatmap
        ax = axes[1, 0]
        self.create_correlation_matrix(ax, genes_to_plot)
        
        # Plot 4: Distribution comparison
        ax = axes[1, 1]
        self.create_distribution_comparison(ax)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "standardized_normalization_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Visualization saved: {plot_path}")
    
    def create_correlation_matrix(self, ax, common_genes):
        """Create correlation matrix between datasets"""
        dataset_names = list(self.control_profiles.keys())
        n_datasets = len(dataset_names)
        
        # Create expression matrix for correlation
        expression_matrix = np.zeros((n_datasets, len(common_genes)))
        
        for i, dataset_name in enumerate(dataset_names):
            profile_data = self.control_profiles[dataset_name]
            gene_names = list(profile_data['gene_names'])
            normalized_profile = profile_data['normalized_profile']
            
            for j, gene in enumerate(common_genes):
                if gene in gene_names:
                    gene_idx = gene_names.index(gene)
                    expression_matrix[i, j] = normalized_profile[gene_idx]
        
        # Calculate correlation
        correlation_matrix = np.corrcoef(expression_matrix)
        
        # Plot heatmap
        im = ax.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        
        # Add labels
        ax.set_xticks(range(n_datasets))
        ax.set_yticks(range(n_datasets))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.set_yticklabels(dataset_names)
        
        # Add correlation values
        for i in range(n_datasets):
            for j in range(n_datasets):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight="bold")
        
        ax.set_title(f'Dataset Correlation Matrix\n({len(common_genes):,} Common Genes)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    
    def create_distribution_comparison(self, ax):
        """Create distribution comparison plot"""
        dataset_names = list(self.control_profiles.keys())
        
        for i, dataset_name in enumerate(dataset_names):
            profile_data = self.control_profiles[dataset_name]
            normalized_profile = profile_data['normalized_profile']
            
            # Remove zeros for better visualization
            nonzero_values = normalized_profile[normalized_profile > self.min_expression_threshold]
            
            if len(nonzero_values) > 0:
                ax.hist(nonzero_values, bins=50, alpha=0.6, label=dataset_name, 
                       density=True, color=plt.cm.Set1(i))
        
        ax.set_xlabel('Normalized Expression')
        ax.set_ylabel('Density')
        ax.set_title('Normalized Expression Distributions')
        ax.legend()
        ax.set_xlim(0, 5)  # Focus on reasonable range
    
    def save_results(self, common_genes):
        """
        Save comprehensive results
        """
        print(f"\n💾 Saving results...")
        
        # Save combined profiles
        combined_results = {
            'control_profiles': self.control_profiles,
            'normalization_stats': self.normalization_stats,
            'common_genes': sorted(list(common_genes)),
            'n_common_genes': len(common_genes),
            'method': 'median_nonzero_standardized_genes',
            'min_expression_threshold': self.min_expression_threshold
        }
        
        results_path = self.output_dir / "standardized_normalization_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(combined_results, f)
        
        print(f"   ✅ Combined results: {results_path}")
        
        # Save common gene profiles as CSV
        if common_genes:
            profile_dict = {'gene': sorted(list(common_genes))}
            
            for dataset_name, profile_data in self.control_profiles.items():
                gene_names = list(profile_data['gene_names'])
                raw_profile = profile_data['raw_profile']
                normalized_profile = profile_data['normalized_profile']
                
                raw_values = []
                norm_values = []
                
                for gene in profile_dict['gene']:
                    if gene in gene_names:
                        gene_idx = gene_names.index(gene)
                        raw_values.append(raw_profile[gene_idx])
                        norm_values.append(normalized_profile[gene_idx])
                    else:
                        raw_values.append(0)
                        norm_values.append(0)
                
                profile_dict[f'{dataset_name}_raw'] = raw_values
                profile_dict[f'{dataset_name}_normalized'] = norm_values
            
            profiles_df = pd.DataFrame(profile_dict)
            csv_path = self.output_dir / "common_gene_profiles_standardized.csv"
            profiles_df.to_csv(csv_path, index=False)
            
            print(f"   ✅ Common gene profiles: {csv_path}")
        
        # Create summary report
        summary_path = self.output_dir / "normalization_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("STANDARDIZED DATASET NORMALIZATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Datasets processed: {len(self.control_profiles)}\n")
            f.write(f"Common genes found: {len(common_genes):,}\n")
            f.write(f"Normalization method: Median non-zero division\n")
            f.write(f"Gene names: Standardized symbols\n\n")
            
            f.write("Dataset Statistics:\n")
            f.write("-" * 30 + "\n")
            for dataset_name, stats in self.normalization_stats.items():
                f.write(f"\n{dataset_name}:\n")
                f.write(f"  Control cells: {stats['n_control_cells']:,}\n")
                f.write(f"  Total genes: {stats['n_genes']:,}\n")
                f.write(f"  Non-zero genes: {stats['n_nonzero_genes']:,}\n")
                f.write(f"  Median non-zero: {stats['median_nonzero_raw']:.4f}\n")
                f.write(f"  Mean normalized: {stats['mean_normalized']:.4f}\n")
            
            f.write(f"\nKey Improvements:\n")
            f.write("- Gene names standardized across all datasets\n")
            f.write(f"- {len(common_genes):,} genes common to all datasets\n")
            f.write("- Cross-dataset comparisons now meaningful\n")
            f.write("- Perturbation targets match gene names\n")
        
        print(f"   ✅ Summary report: {summary_path}")
    
    def run_standardized_normalization(self):
        """
        Main function to run normalization on standardized datasets
        """
        print("🧬 STANDARDIZED DATASET NORMALIZATION")
        print("=" * 60)
        print("🎯 Using gene-name-standardized datasets")
        print("📊 Enabling proper cross-dataset analysis")
        print("=" * 60)
        
        # Find standardized datasets
        datasets = self.find_standardized_datasets()
        
        if not datasets:
            print("❌ No standardized datasets found!")
            print("\n💡 SOLUTION:")
            print("1. Run: python gene_id_converter.py")
            print("2. This will create standardized datasets")
            print("3. Then re-run this normalization")
            return False
        
        # Process each dataset
        successful_datasets = []
        
        for dataset_path, dataset_name in datasets:
            success = self.process_dataset(dataset_path, dataset_name)
            if success:
                successful_datasets.append(dataset_name)
        
        if not successful_datasets:
            print("❌ No datasets were successfully processed!")
            return False
        
        print(f"\n🎉 Successfully processed {len(successful_datasets)} datasets!")
        
        # Analyze gene overlap
        common_genes = self.analyze_gene_overlap()
        
        # Create visualizations
        self.create_visualizations(common_genes)
        
        # Save results
        self.save_results(common_genes)
        
        print(f"\n🎉 STANDARDIZED NORMALIZATION COMPLETE!")
        print("=" * 60)
        print(f"✅ Processed {len(successful_datasets)} datasets")
        print(f"✅ Found {len(common_genes):,} common genes")
        print(f"✅ Results saved in: {self.output_dir}")
        
        if len(common_genes) > 5000:
            print(f"\n🌟 EXCELLENT! {len(common_genes):,} common genes")
            print("   ✅ Cross-dataset analysis will work very well")
        elif len(common_genes) > 1000:
            print(f"\n✅ GOOD! {len(common_genes):,} common genes")
            print("   ✅ Cross-dataset analysis should work well")
        else:
            print(f"\n⚠️ Only {len(common_genes):,} common genes found")
        
        return True


def main():
    """
    Main function
    """
    normalizer = StandardizedDatasetNormalizer()
    success = normalizer.run_standardized_normalization()
    
    if success:
        print(f"\n🔧 NEXT STEPS:")
        print("1. Use results in standardized_normalization_results/")
        print("2. Extract embeddings for perturbation targets")
        print("3. Perform cross-dataset comparisons")
        print("4. Build predictive models")

if __name__ == "__main__":
    main()