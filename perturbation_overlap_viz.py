import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3
import itertools
from pathlib import Path
import h5py
import scanpy as sc
from typing import Dict, List, Set, Tuple
import warnings

warnings.filterwarnings('ignore')


class PerturbationOverlapVisualizer:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.perturbation_sets = {}
        self.vcc_genes = set()
        self.colors = plt.cm.Set3(np.linspace(0, 1, 10))

    def load_perturbation_files(self):
        """Load all perturbation gene files from refined_perturbation_analysis folder"""
        perturbation_dir = self.base_path / "refined_perturbation_analysis"

        if not perturbation_dir.exists():
            print(f"Directory not found: {perturbation_dir}")
            return

        # Map of file patterns to dataset names
        file_mapping = {
            "hepg2_perturbation_genes.txt": "HepG2",
            "jurkat_perturbation_genes.txt": "Jurkat",
            "k562_perturbation_genes.txt": "K562",
            "tf_knockout_perturbation_genes.txt": "TF Knockout",
            "vcc_training_perturbation_genes.txt": "VCC Training"
        }

        for filename, dataset_name in file_mapping.items():
            filepath = perturbation_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    genes = set(line.strip() for line in f if line.strip())
                self.perturbation_sets[dataset_name] = genes
                print(f"Loaded {len(genes)} genes from {dataset_name}")
            else:
                print(f"File not found: {filepath}")

    def load_vcc_submission_genes(self):
        """Extract genes from VCC submission files or load from cached txt files"""
        perturbation_dir = self.base_path / "refined_perturbation_analysis"
        perturbation_dir.mkdir(exist_ok=True)

        vcc_datasets = {
            "vcc_esm2_go_rf_v2_submission": "vcc_esm2_go_rf_v2_perturbation_genes.txt",
            "vcc_esm2_go_rf_submission": "vcc_esm2_go_rf_perturbation_genes.txt",
            "vcc_cross_dataset_submission": "vcc_cross_dataset_perturbation_genes.txt",
            "vcc_esm2_rf_differential_submission": "vcc_esm2_rf_differential_perturbation_genes.txt",
            "vcc_esm2_rf_submission": "vcc_esm2_rf_basic_perturbation_genes.txt"
        }

        for vcc_dir, txt_filename in vcc_datasets.items():
            txt_filepath = perturbation_dir / txt_filename

            # Check if cached file exists and load it
            if txt_filepath.exists():
                with open(txt_filepath, 'r') as f:
                    genes = set(line.strip() for line in f if line.strip())
                if genes:
                    dataset_name = txt_filename.replace('_perturbation_genes.txt', '').replace('vcc_', '').upper()
                    self.perturbation_sets[dataset_name] = genes
                    print(f"Loaded {len(genes)} genes from cached file: {txt_filename}")
                    continue

            # Extract genes from submission files if cache doesn't exist
            dir_path = self.base_path / vcc_dir
            if not dir_path.exists():
                continue

            print(f"Extracting perturbation genes from {vcc_dir}...")
            dataset_genes = set()

            # Try to find h5ad files
            h5ad_files = list(dir_path.glob("*.h5ad"))

            for h5ad_file in h5ad_files:
                try:
                    print(f"  Checking {h5ad_file.name}")
                    adata = sc.read_h5ad(h5ad_file)

                    print(f"    Available obs columns: {list(adata.obs.columns)}")

                    # Check if there are perturbation annotations
                    if 'perturbation' in adata.obs.columns:
                        perturbed_genes = set(adata.obs['perturbation'].dropna().unique())
                        perturbed_genes.discard('control')  # Remove control
                        dataset_genes.update(perturbed_genes)
                        print(f"    Found {len(perturbed_genes)} unique perturbations from 'perturbation' column")

                    # Also check other potential perturbation columns
                    pert_cols = [col for col in adata.obs.columns if 'pert' in col.lower()]
                    for col in pert_cols:
                        perturbed_genes = set(adata.obs[col].dropna().astype(str).unique())
                        perturbed_genes.discard('control')
                        perturbed_genes.discard('nan')
                        perturbed_genes.discard('None')
                        dataset_genes.update(perturbed_genes)
                        print(f"    Found {len(perturbed_genes)} unique perturbations from '{col}' column")

                    # If no perturbation columns found, list all columns for debugging
                    if not any('pert' in col.lower() for col in adata.obs.columns):
                        print(f"    No perturbation columns found. All columns: {list(adata.obs.columns)}")

                except Exception as e:
                    print(f"    Error reading {h5ad_file.name}: {str(e)}")

            # If no genes found in h5ad files, try alternative extraction
            if not dataset_genes:
                dataset_genes = self._extract_genes_from_directories(dir_path)

            # Save extracted genes to txt file for future use
            if dataset_genes:
                with open(txt_filepath, 'w') as f:
                    for gene in sorted(dataset_genes):
                        f.write(f"{gene}\n")

                dataset_name = txt_filename.replace('_perturbation_genes.txt', '').replace('vcc_', '').upper()
                self.perturbation_sets[dataset_name] = dataset_genes
                print(f"  Extracted and saved {len(dataset_genes)} genes to {txt_filename}")
            else:
                print(f"  No perturbation genes found in {vcc_dir}")

        # Also try to extract from validation predictions
        self._extract_from_validation_predictions()

    def _extract_genes_from_directories(self, dir_path: Path) -> Set[str]:
        """Alternative method to extract genes from directory structure"""
        genes = set()

        # Check gene_names.txt files (these might be target genes, not perturbations)
        gene_names_file = dir_path / "gene_names.txt"
        if gene_names_file.exists():
            try:
                with open(gene_names_file, 'r') as f:
                    gene_names = set(line.strip() for line in f if line.strip())
                print(
                    f"    Found {len(gene_names)} genes in gene_names.txt (these are likely target genes, not perturbations)")
                # Note: gene_names.txt typically contains target genes to predict, not perturbation genes
                # So we don't add them to the perturbation set
            except Exception as e:
                print(f"    Error reading gene_names.txt: {str(e)}")

        return genes

    def _extract_from_validation_predictions(self):
        """Extract perturbation info from validation predictions"""
        perturbation_dir = self.base_path / "refined_perturbation_analysis"
        val_pred_txt = perturbation_dir / "validation_predictions_perturbation_genes.txt"

        # Check if cached file exists
        if val_pred_txt.exists():
            with open(val_pred_txt, 'r') as f:
                genes = set(line.strip() for line in f if line.strip())
            if genes:
                self.perturbation_sets["VALIDATION_PREDICTIONS"] = genes
                print(f"Loaded {len(genes)} genes from cached validation predictions file")
                return

        # Extract from validation predictions directory
        val_pred_dir = self.base_path / "outputs" / "validation_predictions"
        val_genes = set()

        if val_pred_dir.exists():
            # Method 1: Try validation summary CSV
            try:
                summary_file = val_pred_dir / "validation_summary.csv"
                if summary_file.exists():
                    df = pd.read_csv(summary_file)
                    if 'perturbation' in df.columns:
                        vcc_perturbed = set(df['perturbation'].dropna().unique())
                        val_genes.update(vcc_perturbed)
                        print(f"  Found {len(vcc_perturbed)} perturbations in validation summary")
            except Exception as e:
                print(f"  Error reading validation summary: {str(e)}")

            # Method 2: Extract from pert_* directory names
            pert_dirs = [d for d in val_pred_dir.iterdir() if d.is_dir() and d.name.startswith('pert_')]
            if pert_dirs:
                for pert_dir in pert_dirs:
                    gene_name = pert_dir.name.replace('pert_', '')
                    val_genes.add(gene_name)
                print(f"  Extracted {len(pert_dirs)} perturbation genes from pert_* directory names")

            # Method 3: Try loading h5ad files in validation directory
            h5ad_files = list(val_pred_dir.glob("*.h5ad"))
            for h5ad_file in h5ad_files:
                try:
                    adata = sc.read_h5ad(h5ad_file)
                    if 'perturbation' in adata.obs.columns:
                        perturbed_genes = set(adata.obs['perturbation'].dropna().unique())
                        perturbed_genes.discard('control')
                        val_genes.update(perturbed_genes)
                        print(f"  Found {len(perturbed_genes)} perturbations in {h5ad_file.name}")
                except Exception as e:
                    print(f"  Error reading {h5ad_file.name}: {str(e)}")

        # Save extracted genes to txt file for future use
        if val_genes:
            with open(val_pred_txt, 'w') as f:
                for gene in sorted(val_genes):
                    f.write(f"{gene}\n")

            self.perturbation_sets["VALIDATION_PREDICTIONS"] = val_genes
            print(f"  Extracted and saved {len(val_genes)} genes to validation_predictions_perturbation_genes.txt")
        else:
            print("  No validation prediction genes found")

    def create_overlap_matrix(self):
        """Create a matrix showing pairwise overlaps"""
        datasets = list(self.perturbation_sets.keys())
        n_datasets = len(datasets)

        # Create overlap matrix
        overlap_matrix = np.zeros((n_datasets, n_datasets))
        overlap_counts = np.zeros((n_datasets, n_datasets))

        for i, dataset1 in enumerate(datasets):
            for j, dataset2 in enumerate(datasets):
                set1 = self.perturbation_sets[dataset1]
                set2 = self.perturbation_sets[dataset2]

                if i == j:
                    overlap_matrix[i, j] = 1.0
                    overlap_counts[i, j] = len(set1)
                else:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    overlap_matrix[i, j] = intersection / union if union > 0 else 0
                    overlap_counts[i, j] = intersection

        return overlap_matrix, overlap_counts, datasets

    def plot_overlap_heatmap(self):
        """Create heatmap visualization of overlaps"""
        if len(self.perturbation_sets) < 2:
            print("Need at least 2 datasets to create overlap visualization")
            return

        overlap_matrix, overlap_counts, datasets = self.create_overlap_matrix()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Jaccard similarity heatmap
        mask = np.triu(np.ones_like(overlap_matrix, dtype=bool), k=1)
        sns.heatmap(overlap_matrix,
                    xticklabels=datasets,
                    yticklabels=datasets,
                    annot=True,
                    fmt='.3f',
                    cmap='YlOrRd',
                    mask=mask,
                    ax=ax1)
        ax1.set_title('Jaccard Similarity (Intersection/Union)')

        # Overlap counts heatmap
        sns.heatmap(overlap_counts.astype(int),
                    xticklabels=datasets,
                    yticklabels=datasets,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    mask=mask,
                    ax=ax2)
        ax2.set_title('Overlap Counts (Number of Shared Genes)')

        plt.tight_layout()
        return fig

    def plot_venn_diagrams(self):
        """Create Venn diagrams for subsets of datasets"""
        datasets = list(self.perturbation_sets.keys())
        n_datasets = len(datasets)

        if n_datasets < 2:
            print("Need at least 2 datasets for Venn diagram")
            return

        # Create multiple subplot configurations based on number of datasets
        if n_datasets == 2:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            sets = [self.perturbation_sets[d] for d in datasets]
            venn2(sets, set_labels=datasets, ax=ax)
            ax.set_title('Gene Overlap Between Datasets')

        elif n_datasets == 3:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            sets = [self.perturbation_sets[d] for d in datasets]
            venn3(sets, set_labels=datasets, ax=ax)
            ax.set_title('Gene Overlap Between Datasets')

        elif n_datasets >= 4:
            # For more than 3 datasets, create pairwise comparisons
            n_pairs = min(6, n_datasets * (n_datasets - 1) // 2)  # Show max 6 pairs
            n_cols = 3
            n_rows = (n_pairs + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            plot_idx = 0
            for i in range(n_datasets):
                for j in range(i + 1, n_datasets):
                    if plot_idx >= n_pairs:
                        break

                    row = plot_idx // n_cols
                    col = plot_idx % n_cols

                    sets = [self.perturbation_sets[datasets[i]],
                            self.perturbation_sets[datasets[j]]]
                    venn2(sets, set_labels=[datasets[i], datasets[j]], ax=axes[row, col])
                    axes[row, col].set_title(f'{datasets[i]} vs {datasets[j]}')

                    plot_idx += 1

                if plot_idx >= n_pairs:
                    break

            # Hide unused subplots
            for idx in range(plot_idx, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].set_visible(False)

        plt.tight_layout()
        return fig

    def create_summary_stats(self):
        """Create summary statistics table"""
        stats_data = []

        for dataset_name, genes in self.perturbation_sets.items():
            stats_data.append({
                'Dataset': dataset_name,
                'Total Genes': len(genes),
                'Sample Genes': ', '.join(sorted(list(genes))[:5]) + ('...' if len(genes) > 5 else '')
            })

        df = pd.DataFrame(stats_data)
        return df

    def find_common_genes(self):
        """Find genes common across all datasets"""
        if not self.perturbation_sets:
            return set()

        common_genes = set.intersection(*self.perturbation_sets.values())
        return common_genes

    def find_unique_genes(self):
        """Find genes unique to each dataset"""
        unique_genes = {}
        all_other_genes = {}

        for dataset_name, genes in self.perturbation_sets.items():
            # Get all genes from other datasets
            other_datasets = {k: v for k, v in self.perturbation_sets.items() if k != dataset_name}
            if other_datasets:
                all_other = set.union(*other_datasets.values())
                unique = genes - all_other
                unique_genes[dataset_name] = unique
                all_other_genes[dataset_name] = all_other
            else:
                unique_genes[dataset_name] = genes

        return unique_genes

    def generate_comprehensive_report(self):
        """Generate a comprehensive overlap analysis report"""
        print("=" * 80)
        print("PERTURBATION GENE OVERLAP ANALYSIS REPORT")
        print("=" * 80)

        # Summary statistics
        print("\n1. DATASET SUMMARY")
        print("-" * 40)
        summary_df = self.create_summary_stats()
        print(summary_df.to_string(index=False))

        # Common genes
        print("\n2. GENES COMMON TO ALL DATASETS")
        print("-" * 40)
        common_genes = self.find_common_genes()
        if common_genes:
            print(f"Found {len(common_genes)} common genes:")
            for gene in sorted(common_genes):
                print(f"  - {gene}")
        else:
            print("No genes are common to all datasets")

        # Unique genes
        print("\n3. GENES UNIQUE TO EACH DATASET")
        print("-" * 40)
        unique_genes = self.find_unique_genes()
        for dataset, genes in unique_genes.items():
            print(f"\n{dataset} unique genes ({len(genes)}):")
            if genes:
                for gene in sorted(list(genes)[:10]):  # Show first 10
                    print(f"  - {gene}")
                if len(genes) > 10:
                    print(f"  ... and {len(genes) - 10} more")
            else:
                print("  No unique genes")

        # Pairwise overlaps
        print("\n4. PAIRWISE OVERLAPS")
        print("-" * 40)
        datasets = list(self.perturbation_sets.keys())
        for i, dataset1 in enumerate(datasets):
            for j, dataset2 in enumerate(datasets[i + 1:], i + 1):
                set1 = self.perturbation_sets[dataset1]
                set2 = self.perturbation_sets[dataset2]
                intersection = set1.intersection(set2)
                union = set1.union(set2)
                jaccard = len(intersection) / len(union) if len(union) > 0 else 0

                print(f"{dataset1} ∩ {dataset2}:")
                print(f"  Shared genes: {len(intersection)}")
                print(f"  Jaccard similarity: {jaccard:.3f}")
                print()

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Loading perturbation gene files...")
        self.load_perturbation_files()

        print("Loading VCC submission genes...")
        self.load_vcc_submission_genes()

        if not self.perturbation_sets:
            print("No perturbation data loaded. Please check file paths.")
            return

        # Generate report
        self.generate_comprehensive_report()

        # Create visualizations
        print("Creating visualizations...")

        # Heatmap
        fig1 = self.plot_overlap_heatmap()
        if fig1:
            plt.figure(fig1.number)
            plt.savefig('perturbation_overlap_heatmap.png', dpi=300, bbox_inches='tight')
            print("Saved: perturbation_overlap_heatmap.png")

        # Venn diagrams
        fig2 = self.plot_venn_diagrams()
        if fig2:
            plt.figure(fig2.number)
            plt.savefig('perturbation_venn_diagrams.png', dpi=300, bbox_inches='tight')
            print("Saved: perturbation_venn_diagrams.png")

        # Save summary to CSV
        summary_df = self.create_summary_stats()
        summary_df.to_csv('perturbation_summary.csv', index=False)
        print("Saved: perturbation_summary.csv")

        plt.show()


# Usage example
if __name__ == "__main__":
    # Initialize the visualizer
    visualizer = PerturbationOverlapVisualizer()

    # Run complete analysis
    visualizer.run_complete_analysis()