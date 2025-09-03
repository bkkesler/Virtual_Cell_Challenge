import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
import os

class VCCPerturbationExtractor:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.output_dir = self.base_path / "refined_perturbation_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_from_submission_directory(self, submission_dir: str, output_filename: str):
        """Extract perturbations from a VCC submission directory"""
        dir_path = self.base_path / submission_dir
        
        if not dir_path.exists():
            print(f"Directory not found: {submission_dir}")
            return set()
            
        print(f"Extracting perturbations from {submission_dir}...")
        all_perturbations = set()
        
        # Find all h5ad files in the directory
        h5ad_files = list(dir_path.glob("*.h5ad"))
        
        for h5ad_file in h5ad_files:
            try:
                print(f"  Checking {h5ad_file.name}...")
                adata = sc.read_h5ad(h5ad_file)
                
                print(f"    Shape: {adata.shape}")
                print(f"    Obs columns: {list(adata.obs.columns)}")
                
                # Look for common perturbation column names
                perturbation_columns = []
                
                # Check for various possible column names
                possible_columns = [
                    'target_gene', 'perturbation', 'pert', 'guide_target', 
                    'gene_target', 'target', 'perturbed_gene', 'knockdown_target'
                ]
                
                for col in possible_columns:
                    if col in adata.obs.columns:
                        perturbation_columns.append(col)
                
                if not perturbation_columns:
                    print(f"    No perturbation columns found")
                    continue
                
                # Extract perturbations from each column
                for col in perturbation_columns:
                    unique_values = set(adata.obs[col].dropna().astype(str).unique())
                    
                    # Remove common control/non-perturbation values
                    control_values = {
                        'control', 'non-targeting', 'non_targeting', 'scramble', 
                        'scrambled', 'negative_control', 'neg_ctrl', 'ctrl',
                        'nan', 'None', 'unknown', ''
                    }
                    
                    perturbations = unique_values - control_values
                    all_perturbations.update(perturbations)
                    
                    print(f"    Column '{col}': {len(unique_values)} unique values, {len(perturbations)} perturbations")
                    print(f"    Sample values: {list(sorted(perturbations))[:5]}")
                    
                    # Show control values found for debugging
                    found_controls = unique_values & control_values
                    if found_controls:
                        print(f"    Control values found: {found_controls}")
                
            except Exception as e:
                print(f"    Error reading {h5ad_file.name}: {str(e)}")
        
        # Save extracted perturbations
        if all_perturbations:
            output_path = self.output_dir / output_filename
            with open(output_path, 'w') as f:
                for gene in sorted(all_perturbations):
                    f.write(f"{gene}\n")
            print(f"  Saved {len(all_perturbations)} perturbations to {output_filename}")
        else:
            print(f"  No perturbations found in {submission_dir}")
            
        return all_perturbations
    
    def extract_all_vcc_submissions(self):
        """Extract perturbations from all VCC submission directories"""
        vcc_submissions = {
            "vcc_esm2_go_rf_v2_submission": "vcc_esm2_go_rf_v2_perturbation_genes.txt",
            "vcc_esm2_go_rf_submission": "vcc_esm2_go_rf_perturbation_genes.txt",
            "vcc_cross_dataset_submission": "vcc_cross_dataset_perturbation_genes.txt",
            "vcc_esm2_rf_differential_submission": "vcc_esm2_rf_differential_perturbation_genes.txt",
            "vcc_esm2_rf_submission": "vcc_esm2_rf_basic_perturbation_genes.txt"
        }
        
        all_results = {}
        
        for submission_dir, output_file in vcc_submissions.items():
            perturbations = self.extract_from_submission_directory(submission_dir, output_file)
            if perturbations:
                dataset_name = output_file.replace('_perturbation_genes.txt', '').replace('vcc_', '').upper()
                all_results[dataset_name] = perturbations
        
        return all_results
    
    def extract_from_validation_predictions(self):
        """Extract perturbations from validation predictions directory"""
        output_file = "validation_predictions_perturbation_genes.txt"
        val_pred_dir = self.base_path / "outputs" / "validation_predictions"
        
        if not val_pred_dir.exists():
            print("Validation predictions directory not found")
            return set()
        
        print("Extracting from validation predictions...")
        all_perturbations = set()
        
        # Method 1: Extract from pert_* directory names
        pert_dirs = [d for d in val_pred_dir.iterdir() if d.is_dir() and d.name.startswith('pert_')]
        
        if pert_dirs:
            for pert_dir in pert_dirs:
                gene_name = pert_dir.name.replace('pert_', '')
                all_perturbations.add(gene_name)
            print(f"  Found {len(pert_dirs)} perturbation directories")
        
        # Method 2: Check validation summary files
        try:
            summary_file = val_pred_dir / "validation_summary.csv"
            if summary_file.exists():
                df = pd.read_csv(summary_file)
                print(f"  Validation summary columns: {list(df.columns)}")
                
                # Look for perturbation columns
                pert_columns = [col for col in df.columns if any(term in col.lower() for term in ['pert', 'target', 'gene'])]
                
                for col in pert_columns:
                    unique_vals = set(df[col].dropna().astype(str).unique())
                    control_vals = {'control', 'non-targeting', 'nan', 'None'}
                    perturbations = unique_vals - control_vals
                    all_perturbations.update(perturbations)
                    print(f"  Column '{col}': {len(perturbations)} perturbations")
                    
        except Exception as e:
            print(f"  Error reading validation summary: {str(e)}")
        
        # Method 3: Check h5ad files in validation directory
        h5ad_files = list(val_pred_dir.glob("*.h5ad"))
        for h5ad_file in h5ad_files:
            try:
                print(f"  Checking {h5ad_file.name}...")
                adata = sc.read_h5ad(h5ad_file)
                
                # Look for perturbation information
                for col in adata.obs.columns:
                    if any(term in col.lower() for term in ['pert', 'target', 'gene']):
                        unique_vals = set(adata.obs[col].dropna().astype(str).unique())
                        control_vals = {'control', 'non-targeting', 'nan', 'None'}
                        perturbations = unique_vals - control_vals
                        all_perturbations.update(perturbations)
                        print(f"    Column '{col}': {len(perturbations)} perturbations")
                        
            except Exception as e:
                print(f"  Error reading {h5ad_file.name}: {str(e)}")
        
        # Save results
        if all_perturbations:
            output_path = self.output_dir / output_file
            with open(output_path, 'w') as f:
                for gene in sorted(all_perturbations):
                    f.write(f"{gene}\n")
            print(f"  Saved {len(all_perturbations)} perturbations to {output_file}")
        
        return all_perturbations
    
    def create_comprehensive_summary(self, all_results):
        """Create a summary of all extracted perturbations"""
        summary_file = self.output_dir / "extraction_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("VCC Perturbation Extraction Summary\n")
            f.write("=" * 40 + "\n\n")
            
            total_datasets = len(all_results)
            total_unique_genes = len(set.union(*all_results.values()) if all_results else set())
            
            f.write(f"Total datasets processed: {total_datasets}\n")
            f.write(f"Total unique perturbations: {total_unique_genes}\n\n")
            
            for dataset_name, perturbations in all_results.items():
                f.write(f"{dataset_name}:\n")
                f.write(f"  Count: {len(perturbations)}\n")
                f.write(f"  Sample genes: {', '.join(sorted(list(perturbations))[:5])}\n")
                if len(perturbations) > 5:
                    f.write(f"  ... and {len(perturbations) - 5} more\n")
                f.write("\n")
            
            # Find overlaps
            if len(all_results) > 1:
                f.write("Overlaps:\n")
                datasets = list(all_results.keys())
                for i, dataset1 in enumerate(datasets):
                    for dataset2 in datasets[i+1:]:
                        overlap = all_results[dataset1] & all_results[dataset2]
                        f.write(f"  {dataset1} ∩ {dataset2}: {len(overlap)} genes\n")
        
        print(f"Summary saved to {summary_file}")
    
    def run_extraction(self):
        """Run the complete extraction process"""
        print("VCC Perturbation Extraction Starting...")
        print("=" * 50)
        
        # Extract from VCC submissions
        vcc_results = self.extract_all_vcc_submissions()
        
        # Extract from validation predictions
        val_perturbations = self.extract_from_validation_predictions()
        if val_perturbations:
            vcc_results["VALIDATION_PREDICTIONS"] = val_perturbations
        
        # Create summary
        if vcc_results:
            self.create_comprehensive_summary(vcc_results)
            print(f"\nExtraction complete! Found perturbations in {len(vcc_results)} datasets.")
        else:
            print("\nNo perturbations extracted. Check your file paths and data structure.")
        
        return vcc_results

if __name__ == "__main__":
    extractor = VCCPerturbationExtractor()
    results = extractor.run_extraction()