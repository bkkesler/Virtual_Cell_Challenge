import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import os

def diagnostic_visualization():
    """
    Diagnostic version with detailed error checking and path verification.
    """
    print("🔍 DIAGNOSTIC VISUALIZATION SCRIPT")
    print("=" * 50)
    
    # Check current working directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Define results directory
    results_dir = Path("median_nonzero_features")
    print(f"📁 Target directory: {results_dir.absolute()}")
    print(f"📁 Directory exists: {results_dir.exists()}")
    
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    # Check write permissions
    try:
        test_file = results_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        print("✅ Write permissions confirmed")
    except Exception as e:
        print(f"❌ Write permission error: {e}")
        return
    
    # List current contents
    print(f"\n📂 Current contents of {results_dir}:")
    for item in results_dir.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name} ({item.stat().st_size / 1024:.2f} KB)")
    
    # Load the combined profiles
    combined_file = results_dir / "combined_normalized_profiles.pkl"
    
    if not combined_file.exists():
        print("❌ Combined profiles file not found!")
        return
    
    print(f"\n📂 Loading {combined_file}...")
    try:
        with open(combined_file, 'rb') as f:
            results = pickle.load(f)
        
        control_profiles = results['control_profiles']
        normalization_stats = results['normalization_stats']
        
        print(f"✅ Loaded data for {len(control_profiles)} datasets:")
        for name in control_profiles.keys():
            print(f"   - {name}")
        
    except Exception as e:
        print(f"❌ Error loading combined profiles: {e}")
        return
    
    # Create visualizations with explicit path checking
    create_diagnostic_plots(control_profiles, normalization_stats, results_dir)

def create_diagnostic_plots(control_profiles, normalization_stats, output_dir):
    """
    Create plots with detailed debugging.
    """
    print(f"\n📊 Creating diagnostic plots in: {output_dir.absolute()}")
    
    # Test 1: Simple summary CSV
    print("\n1️⃣ Creating summary CSV...")
    try:
        summary_data = []
        for dataset_name, stats in normalization_stats.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Control_Cells': stats['n_control_cells'],
                'Total_Genes': stats['n_genes'],
                'Non_zero_Genes': stats['n_nonzero_genes'],
                'Median_Nonzero_Raw': round(stats['median_nonzero_raw'], 4),
                'Mean_Raw': round(stats['mean_raw'], 4),
                'Mean_Normalized': round(stats['mean_normalized'], 4)
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = output_dir / "normalization_summary.csv"
        
        print(f"   📝 Saving to: {csv_path.absolute()}")
        summary_df.to_csv(csv_path, index=False)
        
        if csv_path.exists():
            print(f"   ✅ CSV created successfully ({csv_path.stat().st_size} bytes)")
            print(f"   📋 Preview:\n{summary_df.to_string(index=False)}")
        else:
            print("   ❌ CSV file not found after saving!")
            
    except Exception as e:
        print(f"   ❌ Error creating summary CSV: {e}")
    
    # Test 2: Simple bar plot
    print("\n2️⃣ Creating simple bar plot...")
    try:
        # Find common genes (simplified)
        all_gene_sets = []
        for dataset_name, profile_data in control_profiles.items():
            all_gene_sets.append(set(profile_data['gene_names']))
        
        if len(all_gene_sets) > 1:
            common_genes = set.intersection(*all_gene_sets)
            common_genes = sorted(list(common_genes))
            print(f"   📊 Found {len(common_genes)} common genes")
        else:
            # Fallback: use first dataset genes
            first_dataset = list(control_profiles.keys())[0]
            common_genes = list(control_profiles[first_dataset]['gene_names'][:50])
            print(f"   📊 Using first 50 genes from {first_dataset}")
        
        if len(common_genes) == 0:
            print("   ⚠️ No common genes found, using sample genes")
            common_genes = ['GAPDH', 'ACTB', 'TUBB', 'RPL13A', 'RPS18']
        
        # Create simple plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        dataset_names = list(control_profiles.keys())
        sample_genes = common_genes[:10]  # Just first 10 genes
        
        for i, dataset_name in enumerate(dataset_names):
            profile_data = control_profiles[dataset_name]
            gene_names = profile_data['gene_names']
            normalized_profile = profile_data['normalized_profile']
            
            gene_values = []
            for gene in sample_genes:
                if gene in gene_names:
                    gene_idx = list(gene_names).index(gene)
                    gene_values.append(normalized_profile[gene_idx])
                else:
                    gene_values.append(0)
            
            x_positions = np.arange(len(sample_genes)) + i * 0.8 / len(dataset_names)
            ax.bar(x_positions, gene_values, width=0.8/len(dataset_names), 
                   label=dataset_name, alpha=0.7)
        
        ax.set_xlabel('Genes')
        ax.set_ylabel('Normalized Expression')
        ax.set_title('Sample Gene Expression Comparison')
        ax.set_xticks(np.arange(len(sample_genes)))
        ax.set_xticklabels(sample_genes, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        plot_path = output_dir / "diagnostic_barplot.png"
        print(f"   📊 Saving plot to: {plot_path.absolute()}")
        
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        if plot_path.exists():
            print(f"   ✅ Plot created successfully ({plot_path.stat().st_size / 1024:.2f} KB)")
        else:
            print("   ❌ Plot file not found after saving!")
            
    except Exception as e:
        print(f"   ❌ Error creating bar plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check final directory contents
    print(f"\n3️⃣ Final directory check:")
    try:
        for item in output_dir.iterdir():
            if item.suffix in ['.png', '.csv']:
                print(f"   ✅ {item.name} ({item.stat().st_size / 1024:.2f} KB)")
    except Exception as e:
        print(f"   ❌ Error listing files: {e}")

if __name__ == "__main__":
    diagnostic_visualization()
