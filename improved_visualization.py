import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from collections import Counter

def create_comprehensive_visualizations():
    """
    Create comprehensive visualizations handling datasets with different gene sets.
    """
    print("🎨 CREATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 60)
    
    # Load data
    results_dir = Path("median_nonzero_features")
    combined_file = results_dir / "combined_normalized_profiles.pkl"
    
    with open(combined_file, 'rb') as f:
        results = pickle.load(f)
    
    control_profiles = results['control_profiles']
    normalization_stats = results['normalization_stats']
    
    print(f"📊 Loaded {len(control_profiles)} datasets")
    
    # Create all visualizations
    create_dataset_overview(control_profiles, normalization_stats, results_dir)
    create_gene_overlap_analysis(control_profiles, results_dir)
    create_expression_distributions(control_profiles, results_dir)
    create_normalization_effectiveness(control_profiles, normalization_stats, results_dir)
    create_detailed_summary_tables(control_profiles, normalization_stats, results_dir)
    
    print("\n✅ ALL VISUALIZATIONS COMPLETED!")

def create_dataset_overview(control_profiles, normalization_stats, output_dir):
    """
    Create overview plots showing dataset characteristics.
    """
    print("\n1️⃣ Creating dataset overview...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Overview and Characteristics', fontsize=16, fontweight='bold')
    
    datasets = list(control_profiles.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))
    
    # Plot 1: Number of genes per dataset
    ax = axes[0, 0]
    gene_counts = [len(control_profiles[ds]['gene_names']) for ds in datasets]
    bars = ax.bar(range(len(datasets)), gene_counts, color=colors, alpha=0.7)
    ax.set_title('Total Genes per Dataset', fontweight='bold')
    ax.set_ylabel('Number of Genes')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([ds.replace('_', '\n') for ds in datasets], rotation=0, fontsize=10)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Control cells per dataset
    ax = axes[0, 1]
    control_counts = [normalization_stats[ds]['n_control_cells'] for ds in datasets]
    bars = ax.bar(range(len(datasets)), control_counts, color=colors, alpha=0.7)
    ax.set_title('Control Cells per Dataset', fontweight='bold')
    ax.set_ylabel('Number of Control Cells')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([ds.replace('_', '\n') for ds in datasets], rotation=0, fontsize=10)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Median raw expression levels
    ax = axes[1, 0]
    median_raw = [normalization_stats[ds]['median_nonzero_raw'] for ds in datasets]
    bars = ax.bar(range(len(datasets)), median_raw, color=colors, alpha=0.7)
    ax.set_title('Median Non-Zero Raw Expression', fontweight='bold')
    ax.set_ylabel('Expression Level')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([ds.replace('_', '\n') for ds in datasets], rotation=0, fontsize=10)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Mean normalized expression
    ax = axes[1, 1]
    mean_norm = [normalization_stats[ds]['mean_normalized'] for ds in datasets]
    bars = ax.bar(range(len(datasets)), mean_norm, color=colors, alpha=0.7)
    ax.set_title('Mean Normalized Expression', fontweight='bold')
    ax.set_ylabel('Normalized Expression')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([ds.replace('_', '\n') for ds in datasets], rotation=0, fontsize=10)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = output_dir / "dataset_overview.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_path.name}")

def create_gene_overlap_analysis(control_profiles, output_dir):
    """
    Analyze gene overlap between datasets.
    """
    print("\n2️⃣ Creating gene overlap analysis...")
    
    datasets = list(control_profiles.keys())
    
    # Create pairwise overlap matrix
    overlap_matrix = np.zeros((len(datasets), len(datasets)))
    
    for i, ds1 in enumerate(datasets):
        genes1 = set(control_profiles[ds1]['gene_names'])
        for j, ds2 in enumerate(datasets):
            genes2 = set(control_profiles[ds2]['gene_names'])
            if i == j:
                overlap_matrix[i, j] = len(genes1)
            else:
                overlap = len(genes1.intersection(genes2))
                overlap_matrix[i, j] = overlap
    
    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create labels for better readability
    short_labels = [ds.replace('_', '\n').replace('264667_', '').replace('.sng.guides.full.ct', '') 
                   for ds in datasets]
    
    im = ax.imshow(overlap_matrix, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Overlapping Genes', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(len(datasets)))
    ax.set_yticks(range(len(datasets)))
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.set_yticklabels(short_labels)
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            value = int(overlap_matrix[i, j])
            color = 'white' if value > overlap_matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{value:,}', ha='center', va='center', 
                   color=color, fontweight='bold')
    
    ax.set_title('Gene Overlap Between Datasets', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    plot_path = output_dir / "gene_overlap_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_path.name}")
    
    # Create overlap summary table
    overlap_df = pd.DataFrame(overlap_matrix, 
                             index=datasets, 
                             columns=datasets)
    overlap_path = output_dir / "gene_overlap_matrix.csv"
    overlap_df.to_csv(overlap_path)
    print(f"   ✅ Saved: {overlap_path.name}")

def create_expression_distributions(control_profiles, output_dir):
    """
    Create expression distribution comparisons.
    """
    print("\n3️⃣ Creating expression distributions...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Expression Distribution Analysis', fontsize=16, fontweight='bold')
    
    datasets = list(control_profiles.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(datasets)))
    
    # Plot 1: Raw expression distributions (log scale)
    ax = axes[0, 0]
    for i, dataset in enumerate(datasets):
        profile_data = control_profiles[dataset]
        raw_profile = profile_data['raw_profile']
        nonzero_values = raw_profile[raw_profile > 0.01]
        
        if len(nonzero_values) > 100:
            # Sample for performance
            sample_size = min(10000, len(nonzero_values))
            nonzero_values = np.random.choice(nonzero_values, sample_size, replace=False)
        
        ax.hist(np.log10(nonzero_values + 1e-10), bins=50, alpha=0.6, 
               label=dataset.replace('_', ' '), color=colors[i], density=True)
    
    ax.set_xlabel('Log10(Raw Expression + 1e-10)')
    ax.set_ylabel('Density')
    ax.set_title('Raw Expression Distributions')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Normalized expression distributions
    ax = axes[0, 1]
    for i, dataset in enumerate(datasets):
        profile_data = control_profiles[dataset]
        norm_profile = profile_data['normalized_profile']
        nonzero_values = norm_profile[norm_profile > 0.01]
        
        if len(nonzero_values) > 100:
            sample_size = min(10000, len(nonzero_values))
            nonzero_values = np.random.choice(nonzero_values, sample_size, replace=False)
        
        ax.hist(nonzero_values, bins=50, alpha=0.6, 
               label=dataset.replace('_', ' '), color=colors[i], density=True)
    
    ax.set_xlabel('Normalized Expression')
    ax.set_ylabel('Density')
    ax.set_title('Normalized Expression Distributions')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Box plots of raw expression
    ax = axes[1, 0]
    raw_data = []
    labels = []
    for dataset in datasets:
        profile_data = control_profiles[dataset]
        raw_profile = profile_data['raw_profile']
        nonzero_values = raw_profile[raw_profile > 0]
        
        if len(nonzero_values) > 1000:
            # Sample for performance
            nonzero_values = np.random.choice(nonzero_values, 1000, replace=False)
        
        raw_data.append(np.log10(nonzero_values + 1e-10))
        labels.append(dataset.replace('_', '\n'))
    
    bp = ax.boxplot(raw_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Log10(Raw Expression)')
    ax.set_title('Raw Expression Distribution (Box Plots)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 4: Box plots of normalized expression
    ax = axes[1, 1]
    norm_data = []
    for dataset in datasets:
        profile_data = control_profiles[dataset]
        norm_profile = profile_data['normalized_profile']
        nonzero_values = norm_profile[norm_profile > 0]
        
        if len(nonzero_values) > 1000:
            nonzero_values = np.random.choice(nonzero_values, 1000, replace=False)
        
        norm_data.append(nonzero_values)
    
    bp = ax.boxplot(norm_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Normalized Expression')
    ax.set_title('Normalized Expression Distribution (Box Plots)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    plot_path = output_dir / "expression_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_path.name}")

def create_normalization_effectiveness(control_profiles, normalization_stats, output_dir):
    """
    Evaluate normalization effectiveness.
    """
    print("\n4️⃣ Creating normalization effectiveness analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Normalization Effectiveness Analysis', fontsize=16, fontweight='bold')
    
    datasets = list(control_profiles.keys())
    
    # Calculate coefficients of variation
    raw_cvs = []
    norm_cvs = []
    
    for dataset in datasets:
        profile_data = control_profiles[dataset]
        raw_profile = profile_data['raw_profile']
        norm_profile = profile_data['normalized_profile']
        
        # CV for non-zero values
        raw_nonzero = raw_profile[raw_profile > 0]
        norm_nonzero = norm_profile[norm_profile > 0]
        
        raw_cv = np.std(raw_nonzero) / np.mean(raw_nonzero) if len(raw_nonzero) > 0 else 0
        norm_cv = np.std(norm_nonzero) / np.mean(norm_nonzero) if len(norm_nonzero) > 0 else 0
        
        raw_cvs.append(raw_cv)
        norm_cvs.append(norm_cv)
    
    # Plot 1: CV comparison
    ax = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, raw_cvs, width, label='Raw', alpha=0.7, color='lightcoral')
    bars2 = ax.bar(x + width/2, norm_cvs, width, label='Normalized', alpha=0.7, color='lightblue')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('CV Comparison: Raw vs Normalized')
    ax.set_xticks(x)
    ax.set_xticklabels([ds.replace('_', '\n') for ds in datasets], rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Before vs After normalization scatter
    ax = axes[0, 1]
    
    before_means = [normalization_stats[ds]['mean_raw'] for ds in datasets]
    after_means = [normalization_stats[ds]['mean_normalized'] for ds in datasets]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))
    for i, dataset in enumerate(datasets):
        ax.scatter(before_means[i], after_means[i], s=100, alpha=0.7, 
                  color=colors[i], label=dataset.replace('_', ' '))
    
    ax.set_xlabel('Mean Raw Expression')
    ax.set_ylabel('Mean Normalized Expression')
    ax.set_title('Mean Expression: Before vs After')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Normalization factor by dataset
    ax = axes[1, 0]
    norm_factors = [normalization_stats[ds]['median_nonzero_raw'] for ds in datasets]
    
    bars = ax.bar(range(len(datasets)), norm_factors, color=colors, alpha=0.7)
    ax.set_title('Normalization Factors (Median Non-Zero Raw)')
    ax.set_ylabel('Normalization Factor')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([ds.replace('_', '\n') for ds in datasets], rotation=45, ha='right')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Dynamic range comparison
    ax = axes[1, 1]
    
    raw_ranges = []
    norm_ranges = []
    
    for dataset in datasets:
        raw_max = normalization_stats[dataset]['max_raw']
        norm_max = normalization_stats[dataset]['max_normalized']
        raw_ranges.append(raw_max)
        norm_ranges.append(norm_max)
    
    x = np.arange(len(datasets))
    bars1 = ax.bar(x - width/2, raw_ranges, width, label='Raw Max', alpha=0.7, color='salmon')
    bars2 = ax.bar(x + width/2, norm_ranges, width, label='Normalized Max', alpha=0.7, color='skyblue')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Maximum Expression')
    ax.set_title('Dynamic Range: Raw vs Normalized')
    ax.set_xticks(x)
    ax.set_xticklabels([ds.replace('_', '\n') for ds in datasets], rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    plot_path = output_dir / "normalization_effectiveness.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_path.name}")

def create_detailed_summary_tables(control_profiles, normalization_stats, output_dir):
    """
    Create detailed summary tables.
    """
    print("\n5️⃣ Creating detailed summary tables...")
    
    # Enhanced summary statistics
    detailed_stats = []
    
    for dataset_name, stats in normalization_stats.items():
        profile_data = control_profiles[dataset_name]
        raw_profile = profile_data['raw_profile']
        norm_profile = profile_data['normalized_profile']
        
        # Calculate additional statistics
        raw_nonzero = raw_profile[raw_profile > 0]
        norm_nonzero = norm_profile[norm_profile > 0]
        
        detailed_stats.append({
            'Dataset': dataset_name,
            'Control_Cells': stats['n_control_cells'],
            'Total_Genes': stats['n_genes'],
            'Non_Zero_Genes': stats['n_nonzero_genes'],
            'Sparsity_%': round((1 - stats['n_nonzero_genes'] / stats['n_genes']) * 100, 2),
            'Raw_Median': round(stats['median_nonzero_raw'], 4),
            'Raw_Mean': round(stats['mean_raw'], 4),
            'Raw_Std': round(np.std(raw_nonzero), 4) if len(raw_nonzero) > 0 else 0,
            'Raw_CV': round(np.std(raw_nonzero) / np.mean(raw_nonzero), 4) if len(raw_nonzero) > 0 else 0,
            'Norm_Mean': round(stats['mean_normalized'], 4),
            'Norm_Std': round(np.std(norm_nonzero), 4) if len(norm_nonzero) > 0 else 0,
            'Norm_CV': round(np.std(norm_nonzero) / np.mean(norm_nonzero), 4) if len(norm_nonzero) > 0 else 0,
            'Raw_Max': round(stats['max_raw'], 2),
            'Norm_Max': round(stats['max_normalized'], 2),
            'Normalization_Factor': round(stats['median_nonzero_raw'], 4)
        })
    
    detailed_df = pd.DataFrame(detailed_stats)
    
    # Save detailed summary
    detailed_path = output_dir / "detailed_normalization_summary.csv"
    detailed_df.to_csv(detailed_path, index=False)
    print(f"   ✅ Saved: {detailed_path.name}")
    
    # Create a comparison table focusing on normalization effectiveness
    effectiveness_data = []
    for i, dataset in enumerate(detailed_df['Dataset']):
        row = detailed_df.iloc[i]
        cv_reduction = ((row['Raw_CV'] - row['Norm_CV']) / row['Raw_CV'] * 100) if row['Raw_CV'] > 0 else 0
        
        effectiveness_data.append({
            'Dataset': dataset,
            'CV_Raw': row['Raw_CV'],
            'CV_Normalized': row['Norm_CV'],
            'CV_Reduction_%': round(cv_reduction, 2),
            'Mean_Shift': round(row['Norm_Mean'] / row['Raw_Mean'], 3) if row['Raw_Mean'] > 0 else 0,
            'Normalization_Quality': 'Good' if cv_reduction > 0 else 'Poor'
        })
    
    effectiveness_df = pd.DataFrame(effectiveness_data)
    effectiveness_path = output_dir / "normalization_effectiveness_summary.csv"
    effectiveness_df.to_csv(effectiveness_path, index=False)
    print(f"   ✅ Saved: {effectiveness_path.name}")
    
    # Print summary to console
    print(f"\n📋 NORMALIZATION EFFECTIVENESS SUMMARY:")
    print(effectiveness_df.to_string(index=False))

if __name__ == "__main__":
    create_comprehensive_visualizations()
