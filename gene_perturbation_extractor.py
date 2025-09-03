#!/usr/bin/env python3
"""
Refined Gene Perturbation Extractor

Based on the dataset preview, this script extracts perturbation genes from specific
columns in each dataset type, using the correct parsing logic for each format.
"""

import os
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
import re
from pathlib import Path
from collections import defaultdict
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
sc.settings.verbosity = 0


class RefinedGenePerturbationExtractor:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.results = {}
        self.all_perturbations = set()

    def find_h5ad_files(self):
        """Find h5ad files, including large ones now that we know what to look for"""
        h5ad_files = []

        # Key datasets based on the preview
        target_files = [
            # Training datasets
            "data/raw/single_cell_rnaseq/adata_Training.h5ad",
            "data/raw/single_cell_rnaseq/adata_Training_subset.h5ad",
            "data/processed/normalized/VCC_Training_Subset_normalized.h5ad",

            # HepG2 dataset
            "data/raw/single_cell_rnaseq/GSE264667_hepg2_raw_singlecell_01.h5ad",
            "data/processed/normalized/264667_hepg2_normalized.h5ad",
            "standardized_datasets/264667_hepg2_standardized.h5ad",

            # Jurkat dataset
            "data/raw/single_cell_rnaseq/GSE264667_jurkat_raw_singlecell_01.h5ad",
            "standardized_datasets/264667_jurkat_standardized.h5ad",

            # K562 dataset
            "data/raw/single_cell_rnaseq/K562_essential_raw_singlecell_01.h5ad",
            "standardized_datasets/K562_essential_standardized.h5ad",

            # TF knockout dataset
            "data/raw/single_cell_rnaseq/GSE274751_tfko.sng.guides.full.ct.h5ad",
            "data/processed/normalized/274751_tfko.sng.guides.full.ct_normalized.h5ad",
            "standardized_datasets/274751_tfko.sng.guides.full.ct_standardized.h5ad"
        ]

        for file_rel_path in target_files:
            file_path = self.root_dir / file_rel_path
            if file_path.exists():
                h5ad_files.append(file_path)

        return sorted(h5ad_files)

    def extract_gene_from_guide_string(self, guide_string):
        """Extract gene name from guide format strings"""
        if pd.isna(guide_string) or guide_string == 'nan':
            return None

        guide_str = str(guide_string)

        # Pattern 1: "GENE.number.sequence" (e.g., "YEATS4.69370716.CAGCTTTAGCAAATGATACA")
        match1 = re.match(r'^([A-Z0-9]+)\.[\d]+\.', guide_str)
        if match1:
            return match1.group(1)

        # Pattern 2: "number_GENE_transcript_ensembl" (e.g., "8832_TFAM_P1P2_ENSG00000108064")
        match2 = re.match(r'^\d+_([A-Z0-9]+)_', guide_str)
        if match2:
            return match2.group(1)

        # Pattern 3: "GENE_direction_position" (e.g., "TFAM_+_60145205.23-P1P2|TFAM_-_60145223.23-P1P2")
        match3 = re.match(r'^([A-Z0-9]+)_[+-]_', guide_str)
        if match3:
            return match3.group(1)

        return None

    def extract_perturbations_from_dataset(self, file_path):
        """Extract perturbations from a specific dataset"""
        try:
            print(f"Processing: {file_path.name}")

            # Load the data
            adata = sc.read_h5ad(file_path)

            result = {
                'file_path': str(file_path.relative_to(self.root_dir)),
                'dataset_type': self.identify_dataset_type(file_path),
                'n_obs': adata.n_obs,
                'n_vars': adata.n_vars,
                'perturbations': set(),
                'perturbation_sources': [],
                'metadata_keys': list(adata.obs.columns),
            }

            # Extract based on dataset type
            if 'VCC_Training' in file_path.name or 'adata_Training' in file_path.name:
                result['perturbations'].update(self.extract_vcc_training_genes(adata))
                result['perturbation_sources'].append('target_gene column')

            elif '264667_hepg2' in file_path.name:
                result['perturbations'].update(self.extract_hepg2_genes(adata))
                result['perturbation_sources'].append('gene column')

            elif '264667_jurkat' in file_path.name:
                result['perturbations'].update(self.extract_jurkat_genes(adata))
                result['perturbation_sources'].append('gene column (assumed same as hepg2)')

            elif 'K562_essential' in file_path.name:
                result['perturbations'].update(self.extract_k562_genes(adata))
                result['perturbation_sources'].append('gene column (assumed same as hepg2)')

            elif '274751_tfko' in file_path.name or 'GSE274751' in file_path.name:
                result['perturbations'].update(self.extract_tfko_genes(adata))
                result['perturbation_sources'].append('guide columns')

            # Convert set to sorted list
            result['perturbations'] = sorted(list(result['perturbations']))
            self.all_perturbations.update(result['perturbations'])

            print(f"  Found {len(result['perturbations'])} perturbations")
            if len(result['perturbations']) <= 10:
                print(f"  Genes: {', '.join(result['perturbations'])}")
            else:
                print(f"  First 10 genes: {', '.join(result['perturbations'][:10])}...")

            return result

        except Exception as e:
            print(f"  Error processing {file_path.name}: {str(e)}")
            return {
                'file_path': str(file_path.relative_to(self.root_dir)),
                'error': str(e),
                'perturbations': [],
                'perturbation_sources': [],
                'metadata_keys': []
            }

    def identify_dataset_type(self, file_path):
        """Identify the type of dataset"""
        name = file_path.name.lower()
        if 'vcc_training' in name or 'adata_training' in name:
            return 'VCC_Training'
        elif '264667_hepg2' in name:
            return 'HepG2'
        elif '264667_jurkat' in name:
            return 'Jurkat'
        elif 'k562_essential' in name:
            return 'K562'
        elif '274751_tfko' in name or 'gse274751' in name:
            return 'TF_Knockout'
        else:
            return 'Unknown'

    def extract_vcc_training_genes(self, adata):
        """Extract genes from VCC training dataset"""
        genes = set()

        if 'target_gene' in adata.obs.columns:
            unique_targets = adata.obs['target_gene'].unique()
            for target in unique_targets:
                if pd.notna(target) and str(target).strip():
                    target_str = str(target).strip()
                    # Exclude controls
                    if not any(ctrl in target_str.lower() for ctrl in ['non-targeting', 'control', 'ctrl']):
                        genes.add(target_str)

        return genes

    def extract_hepg2_genes(self, adata):
        """Extract genes from HepG2 dataset"""
        genes = set()

        if 'gene' in adata.obs.columns:
            unique_genes = adata.obs['gene'].unique()
            for gene in unique_genes:
                if pd.notna(gene) and str(gene).strip():
                    gene_str = str(gene).strip()
                    # Exclude controls
                    if not any(ctrl in gene_str.lower() for ctrl in ['non-targeting', 'control', 'ctrl']):
                        genes.add(gene_str)

        return genes

    def extract_jurkat_genes(self, adata):
        """Extract genes from Jurkat dataset (assuming same structure as HepG2)"""
        return self.extract_hepg2_genes(adata)

    def extract_k562_genes(self, adata):
        """Extract genes from K562 dataset (assuming same structure as HepG2)"""
        return self.extract_hepg2_genes(adata)

    def extract_tfko_genes(self, adata):
        """Extract genes from TF knockout dataset"""
        genes = set()

        # Extract from guide columns
        guide_columns = [col for col in adata.obs.columns if 'guide' in col.lower() and '_cov' in col]

        for col in guide_columns:
            unique_guides = adata.obs[col].unique()
            for guide in unique_guides:
                gene = self.extract_gene_from_guide_string(guide)
                if gene and len(gene) > 2:  # Basic filtering
                    genes.add(gene)

        return genes

    def run_analysis(self):
        """Run the complete analysis"""
        print("=" * 80)
        print("REFINED GENE PERTURBATION EXTRACTION")
        print("=" * 80)

        # Find target h5ad files
        h5ad_files = self.find_h5ad_files()
        print(f"\nFound {len(h5ad_files)} target h5ad files to process\n")

        # Process each file
        for file_path in h5ad_files:
            result = self.extract_perturbations_from_dataset(file_path)
            self.results[str(file_path.relative_to(self.root_dir))] = result
            print()

        # Generate summary
        self.generate_summary()

        # Save results
        self.save_results()

    def generate_summary(self):
        """Generate and print summary statistics"""
        print("=" * 80)
        print("SUMMARY BY DATASET TYPE")
        print("=" * 80)

        # Group by dataset type
        by_type = defaultdict(list)
        for file_path, result in self.results.items():
            dataset_type = result.get('dataset_type', 'Unknown')
            by_type[dataset_type].append(result)

        for dataset_type, datasets in by_type.items():
            print(f"\n{dataset_type} DATASETS:")
            print("-" * 50)

            all_genes_for_type = set()
            for dataset in datasets:
                genes = dataset.get('perturbations', [])
                all_genes_for_type.update(genes)

                print(f"  {Path(dataset['file_path']).name}:")
                print(f"    - Observations: {dataset.get('n_obs', 'N/A')}")
                print(f"    - Perturbations: {len(genes)}")
                print(f"    - Sources: {', '.join(dataset.get('perturbation_sources', []))}")

                if len(genes) <= 15:
                    print(f"    - Genes: {', '.join(genes)}")
                else:
                    print(f"    - Sample genes: {', '.join(list(genes)[:15])}...")
                print()

            print(f"  TOTAL UNIQUE GENES FOR {dataset_type}: {len(all_genes_for_type)}")
            if len(all_genes_for_type) <= 30:
                print(f"  All genes: {', '.join(sorted(all_genes_for_type))}")
            else:
                print(f"  Sample genes: {', '.join(sorted(list(all_genes_for_type))[:30])}...")
            print()

        print(f"\nGRAND TOTAL UNIQUE PERTURBATIONS: {len(self.all_perturbations)}")
        print("=" * 80)

    def save_results(self):
        """Save results to files"""
        output_dir = self.root_dir / "refined_perturbation_analysis"
        output_dir.mkdir(exist_ok=True)

        # Save detailed results as JSON
        results_for_json = {}
        for file_path, result in self.results.items():
            # Convert sets to lists for JSON serialization
            result_copy = result.copy()
            if isinstance(result_copy.get('perturbations'), set):
                result_copy['perturbations'] = sorted(list(result_copy['perturbations']))
            results_for_json[file_path] = result_copy

        with open(output_dir / "refined_perturbation_results.json", 'w') as f:
            json.dump(results_for_json, f, indent=2)

        # Save by dataset type
        by_type = defaultdict(set)
        for file_path, result in self.results.items():
            dataset_type = result.get('dataset_type', 'Unknown')
            genes = result.get('perturbations', [])
            by_type[dataset_type].update(genes)

        # Save dataset-specific gene lists
        for dataset_type, genes in by_type.items():
            filename = f"{dataset_type.lower()}_perturbation_genes.txt"
            with open(output_dir / filename, 'w') as f:
                for gene in sorted(genes):
                    f.write(f"{gene}\n")

        # Save all unique perturbations
        with open(output_dir / "all_unique_perturbations.txt", 'w') as f:
            for gene in sorted(self.all_perturbations):
                f.write(f"{gene}\n")

        # Save summary CSV
        summary_data = []
        for file_path, result in self.results.items():
            summary_data.append({
                'file_path': file_path,
                'dataset_type': result.get('dataset_type', 'Unknown'),
                'n_observations': result.get('n_obs', 0),
                'n_variables': result.get('n_vars', 0),
                'n_perturbations': len(result.get('perturbations', [])),
                'perturbation_sources': ', '.join(result.get('perturbation_sources', [])),
                'has_error': 'error' in result
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / "perturbation_summary.csv", index=False)

        print(f"\nResults saved to: {output_dir}")
        print("Files created:")
        print("  - refined_perturbation_results.json (complete analysis)")
        print("  - perturbation_summary.csv (summary table)")
        print("  - all_unique_perturbations.txt (all genes)")
        print("  - *_perturbation_genes.txt (genes by dataset type)")


def main():
    # Set the root directory path
    root_dir = "D:/Virtual_Cell3"  # Adjust this path as needed

    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} not found!")
        print("Please update the root_dir variable in the main() function")
        return

    # Create extractor and run analysis
    extractor = RefinedGenePerturbationExtractor(root_dir)
    extractor.run_analysis()


if __name__ == "__main__":
    main()