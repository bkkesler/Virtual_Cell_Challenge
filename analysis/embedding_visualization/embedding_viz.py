# ESM2 + GO Embedding Visualization for VCC
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pickle
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("🧬 ESM2 + GO EMBEDDING VISUALIZATION")
print("=" * 45)

# Configuration
SAVE_PLOTS = True
OUTPUT_DIR = "./embedding_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# LOAD DATA AND EMBEDDINGS
# =============================================================================

def load_vcc_data():
    """Load VCC training and validation data"""
    vcc_data_dir = "data/raw/single_cell_rnaseq/vcc_data"
    data_path = os.path.join(vcc_data_dir, 'adata_Training.h5ad')
    pert_counts_path = os.path.join(vcc_data_dir, 'pert_counts_Validation.csv')
    
    # Check fallback path
    if not os.path.exists(data_path):
        vcc_data_dir = r"D:\Downloads\vcc_data\vcc_data"
        data_path = os.path.join(vcc_data_dir, 'adata_Training.h5ad')
        pert_counts_path = os.path.join(vcc_data_dir, 'pert_counts_Validation.csv')
    
    if not os.path.exists(data_path):
        print("❌ VCC data not found!")
        return None, None
    
    import scanpy as sc
    adata = sc.read_h5ad(data_path)
    pert_counts = pd.read_csv(pert_counts_path)
    
    # Add non-targeting
    non_targeting_entry = pd.DataFrame({
        'target_gene': ['non-targeting'],
        'n_cells': [1000]
    })
    pert_counts_extended = pd.concat([pert_counts, non_targeting_entry], ignore_index=True)
    
    return adata, pert_counts_extended

def load_esm2_embeddings():
    """Load ESM2 embeddings from available paths"""
    possible_paths = [
        "saved_esm2_go_rf_models/esm2_embeddings_dict.pkl",
        "outputs/state_model_run/embeddings/esm2_embeddings.pt",
        "outputs/state_model_FINAL_SUCCESS/embeddings/esm2_embeddings.pt",
        "saved_esm2_rf_models/esm2_embeddings_dict.pkl"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"📁 Loading ESM2 embeddings from: {path}")
            
            if path.endswith('.pt'):
                embeddings = torch.load(path, weights_only=False)
                esm2_embeddings = {}
                for gene_name, embedding in embeddings.items():
                    if isinstance(embedding, torch.Tensor):
                        esm2_embeddings[gene_name] = embedding.detach().cpu().numpy().astype(np.float32)
                    else:
                        esm2_embeddings[gene_name] = np.array(embedding, dtype=np.float32)
            else:
                with open(path, 'rb') as f:
                    esm2_embeddings = pickle.load(f)
            
            return esm2_embeddings
    
    print("❌ ESM2 embeddings not found!")
    return None

def load_go_embeddings():
    """Load GO embeddings"""
    possible_paths = [
        "saved_esm2_go_rf_models/gene_to_go_mapping.pkl",
        "saved_esm2_go_rf_v2_models/gene_to_go_mapping.pkl",
        "models/embeddings/gene_to_go_mapping.pkl"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"📁 Loading GO embeddings from: {path}")
            with open(path, 'rb') as f:
                go_embeddings = pickle.load(f)
            return go_embeddings
    
    print("⚠️ GO embeddings not found - will create random ones for visualization")
    return None

# =============================================================================
# EMBEDDING COMBINATION FUNCTION
# =============================================================================

def get_combined_embedding(gene_name, esm2_embeddings, go_embeddings, esm2_weight=0.7, go_weight=0.3):
    """Create combined ESM2 + GO embedding"""
    # Get ESM2 embedding
    if gene_name in esm2_embeddings:
        esm2_emb = esm2_embeddings[gene_name]
    else:
        # Fallback: average of all ESM2 embeddings
        esm2_emb = np.mean(list(esm2_embeddings.values()), axis=0)
    
    # Get GO embedding
    if go_embeddings and gene_name in go_embeddings:
        go_emb = go_embeddings[gene_name]
    else:
        # Fallback: random or zero embedding
        go_dim = 256 if go_embeddings is None else list(go_embeddings.values())[0].shape[0]
        go_emb = np.zeros(go_dim, dtype=np.float32)
    
    # Normalize embeddings
    esm2_emb_norm = esm2_emb / (np.linalg.norm(esm2_emb) + 1e-8)
    go_emb_norm = go_emb / (np.linalg.norm(go_emb) + 1e-8)
    
    # Combine with weights
    combined = np.concatenate([
        esm2_weight * esm2_emb_norm,
        go_weight * go_emb_norm
    ])
    
    return combined.astype(np.float32)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_embedding_distribution(embeddings_dict, title, save_path=None):
    """Plot distribution of embedding values"""
    all_values = []
    for emb in embeddings_dict.values():
        all_values.extend(emb.flatten())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(all_values, bins=50, alpha=0.7, density=True)
    ax1.set_xlabel('Embedding Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{title} - Value Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Box plot of norms
    norms = [np.linalg.norm(emb) for emb in embeddings_dict.values()]
    ax2.boxplot(norms)
    ax2.set_ylabel('L2 Norm')
    ax2.set_title(f'{title} - Embedding Norms')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def reduce_dimensions_and_plot(embeddings_matrix, gene_names, gene_types, method='PCA', n_components=2):
    """Reduce dimensionality and create scatter plot"""
    
    if method == 'PCA':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'TSNE':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(gene_names)//4))
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=min(15, len(gene_names)//3))
    else:
        raise ValueError("Method must be 'PCA', 'TSNE', or 'UMAP'")
    
    print(f"🔄 Applying {method} dimensionality reduction...")
    reduced_embeddings = reducer.fit_transform(embeddings_matrix)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'gene': gene_names,
        'type': gene_types
    })
    
    return plot_df, reducer

def create_comprehensive_visualization(embeddings_dict, training_genes, test_genes, embedding_type="Combined"):
    """Create comprehensive embedding visualization"""
    
    # Prepare data
    all_genes = list(embeddings_dict.keys())
    embeddings_matrix = np.array([embeddings_dict[gene] for gene in all_genes])
    
    # Create gene type labels
    gene_types = []
    for gene in all_genes:
        if gene == 'non-targeting':
            gene_types.append('Control')
        elif gene in training_genes:
            gene_types.append('Training')
        elif gene in test_genes:
            gene_types.append('Test')
        else:
            gene_types.append('Other')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: PCA
    plt.subplot(2, 3, 1)
    pca_df, pca_reducer = reduce_dimensions_and_plot(embeddings_matrix, all_genes, gene_types, 'PCA')
    
    # Color map for types
    type_colors = {'Training': '#FF6B6B', 'Test': '#4ECDC4', 'Control': '#45B7D1', 'Other': '#96CEB4'}
    
    for gene_type in pca_df['type'].unique():
        mask = pca_df['type'] == gene_type
        plt.scatter(pca_df[mask]['x'], pca_df[mask]['y'], 
                   label=f'{gene_type} (n={mask.sum()})', 
                   alpha=0.7, s=50, c=type_colors.get(gene_type, 'gray'))
    
    plt.xlabel(f'PC1 ({pca_reducer.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_reducer.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(f'{embedding_type} Embeddings - PCA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: t-SNE (if we have enough genes)
    if len(all_genes) >= 50:
        plt.subplot(2, 3, 2)
        tsne_df, _ = reduce_dimensions_and_plot(embeddings_matrix, all_genes, gene_types, 'TSNE')
        
        for gene_type in tsne_df['type'].unique():
            mask = tsne_df['type'] == gene_type
            plt.scatter(tsne_df[mask]['x'], tsne_df[mask]['y'], 
                       label=f'{gene_type} (n={mask.sum()})', 
                       alpha=0.7, s=50, c=type_colors.get(gene_type, 'gray'))
        
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title(f'{embedding_type} Embeddings - t-SNE')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 3: UMAP (if available)
    try:
        plt.subplot(2, 3, 3)
        umap_df, _ = reduce_dimensions_and_plot(embeddings_matrix, all_genes, gene_types, 'UMAP')
        
        for gene_type in umap_df['type'].unique():
            mask = umap_df['type'] == gene_type
            plt.scatter(umap_df[mask]['x'], umap_df[mask]['y'], 
                       label=f'{gene_type} (n={mask.sum()})', 
                       alpha=0.7, s=50, c=type_colors.get(gene_type, 'gray'))
        
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title(f'{embedding_type} Embeddings - UMAP')
        plt.legend()
        plt.grid(True, alpha=0.3)
    except:
        print("⚠️ UMAP not available")
    
    # Plot 4: Embedding dimension analysis
    plt.subplot(2, 3, 4)
    embedding_dims = [embeddings_dict[gene].shape[0] for gene in all_genes]
    plt.hist(embedding_dims, bins=10, alpha=0.7)
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Count')
    plt.title(f'{embedding_type} - Dimension Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Gene type distribution
    plt.subplot(2, 3, 5)
    type_counts = pd.Series(gene_types).value_counts()
    colors = [type_colors.get(t, 'gray') for t in type_counts.index]
    plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Gene Type Distribution')
    
    # Plot 6: Similarity heatmap (sample)
    plt.subplot(2, 3, 6)
    
    # Sample genes for similarity analysis
    sample_size = min(50, len(all_genes))
    sample_indices = np.random.choice(len(all_genes), sample_size, replace=False)
    sample_embeddings = embeddings_matrix[sample_indices]
    sample_genes = [all_genes[i] for i in sample_indices]
    
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(sample_embeddings)
    
    # Create heatmap
    plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title(f'Similarity Heatmap (Sample of {sample_size} genes)')
    plt.xlabel('Gene Index')
    plt.ylabel('Gene Index')
    
    plt.tight_layout()
    
    if SAVE_PLOTS:
        save_path = os.path.join(OUTPUT_DIR, f'{embedding_type.lower()}_comprehensive_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    
    return pca_df, pca_reducer

def analyze_training_vs_test_separation(embeddings_dict, training_genes, test_genes):
    """Analyze how well training and test genes are separated in embedding space"""
    
    # Get embeddings for training and test genes
    training_embeddings = []
    test_embeddings = []
    
    for gene in training_genes:
        if gene in embeddings_dict:
            training_embeddings.append(embeddings_dict[gene])
    
    for gene in test_genes:
        if gene in embeddings_dict:
            test_embeddings.append(embeddings_dict[gene])
    
    if len(training_embeddings) == 0 or len(test_embeddings) == 0:
        print("⚠️ Insufficient training or test genes for separation analysis")
        return
    
    training_matrix = np.array(training_embeddings)
    test_matrix = np.array(test_embeddings)
    
    # Calculate within-group similarities
    from sklearn.metrics.pairwise import cosine_similarity
    
    train_similarities = cosine_similarity(training_matrix)
    test_similarities = cosine_similarity(test_matrix)
    
    # Calculate between-group similarities
    cross_similarities = cosine_similarity(training_matrix, test_matrix)
    
    # Statistics
    train_mean_sim = np.mean(train_similarities[np.triu_indices_from(train_similarities, k=1)])
    test_mean_sim = np.mean(test_similarities[np.triu_indices_from(test_similarities, k=1)])
    cross_mean_sim = np.mean(cross_similarities)
    
    print(f"\n🔍 EMBEDDING SEPARATION ANALYSIS:")
    print(f"   Training genes available: {len(training_embeddings)}")
    print(f"   Test genes available: {len(test_embeddings)}")
    print(f"   Mean within-training similarity: {train_mean_sim:.4f}")
    print(f"   Mean within-test similarity: {test_mean_sim:.4f}")
    print(f"   Mean cross-group similarity: {cross_mean_sim:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training similarities
    axes[0].hist(train_similarities[np.triu_indices_from(train_similarities, k=1)], 
                bins=30, alpha=0.7, color='red', label='Training')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Within-Training Similarities')
    axes[0].axvline(train_mean_sim, color='darkred', linestyle='--', label=f'Mean: {train_mean_sim:.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Test similarities
    axes[1].hist(test_similarities[np.triu_indices_from(test_similarities, k=1)], 
                bins=30, alpha=0.7, color='blue', label='Test')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Within-Test Similarities')
    axes[1].axvline(test_mean_sim, color='darkblue', linestyle='--', label=f'Mean: {test_mean_sim:.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Cross similarities
    axes[2].hist(cross_similarities.flatten(), bins=30, alpha=0.7, color='green', label='Cross-group')
    axes[2].set_xlabel('Cosine Similarity')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Training vs Test Similarities')
    axes[2].axvline(cross_mean_sim, color='darkgreen', linestyle='--', label=f'Mean: {cross_mean_sim:.3f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if SAVE_PLOTS:
        save_path = os.path.join(OUTPUT_DIR, 'training_test_separation_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("\n📊 Loading data...")
    
    # Load VCC data
    adata, pert_counts = load_vcc_data()
    if adata is None:
        print("❌ Cannot proceed without VCC data")
        return
    
    # Get training and test genes
    training_genes = set(adata.obs['target_gene'].unique())
    test_genes = set(pert_counts['target_gene'].unique())
    
    print(f"✅ Training genes: {len(training_genes)}")
    print(f"✅ Test genes: {len(test_genes)}")
    print(f"✅ Overlap: {len(training_genes.intersection(test_genes))}")
    
    # Load embeddings
    print("\n🧬 Loading embeddings...")
    esm2_embeddings = load_esm2_embeddings()
    if esm2_embeddings is None:
        print("❌ Cannot proceed without ESM2 embeddings")
        return
    
    go_embeddings = load_go_embeddings()
    
    esm2_dim = list(esm2_embeddings.values())[0].shape[0]
    go_dim = 256 if go_embeddings is None else list(go_embeddings.values())[0].shape[0]
    
    print(f"✅ ESM2 embeddings: {len(esm2_embeddings)} genes, {esm2_dim}D")
    if go_embeddings:
        print(f"✅ GO embeddings: {len(go_embeddings)} genes, {go_dim}D")
    else:
        print(f"⚠️ GO embeddings: Using zeros, {go_dim}D")
    
    # Create combined embeddings for all relevant genes
    print("\n🔗 Creating combined embeddings...")
    all_relevant_genes = training_genes.union(test_genes)
    
    # Add non-targeting if not present
    all_relevant_genes.add('non-targeting')
    
    combined_embeddings = {}
    esm2_only_embeddings = {}
    go_only_embeddings = {}
    
    for gene in all_relevant_genes:
        # Combined embedding
        combined_embeddings[gene] = get_combined_embedding(gene, esm2_embeddings, go_embeddings)
        
        # ESM2 only
        if gene in esm2_embeddings:
            esm2_only_embeddings[gene] = esm2_embeddings[gene]
        else:
            esm2_only_embeddings[gene] = np.mean(list(esm2_embeddings.values()), axis=0)
        
        # GO only
        if go_embeddings and gene in go_embeddings:
            go_only_embeddings[gene] = go_embeddings[gene]
        else:
            go_only_embeddings[gene] = np.zeros(go_dim, dtype=np.float32)
    
    print(f"✅ Created combined embeddings for {len(combined_embeddings)} genes")
    
    # Visualizations
    print("\n📈 Creating visualizations...")
    
    # 1. Combined embeddings analysis
    print("\n1️⃣ Combined ESM2 + GO embeddings analysis...")
    create_comprehensive_visualization(combined_embeddings, training_genes, test_genes, "Combined ESM2+GO")
    
    # 2. ESM2 only analysis
    print("\n2️⃣ ESM2 only embeddings analysis...")
    create_comprehensive_visualization(esm2_only_embeddings, training_genes, test_genes, "ESM2 Only")
    
    # 3. GO only analysis (if available)
    if go_embeddings:
        print("\n3️⃣ GO only embeddings analysis...")
        create_comprehensive_visualization(go_only_embeddings, training_genes, test_genes, "GO Only")
    
    # 4. Training vs Test separation analysis
    print("\n4️⃣ Training vs Test separation analysis...")
    print("\nCombined embeddings:")
    analyze_training_vs_test_separation(combined_embeddings, training_genes, test_genes)
    
    print("\nESM2 embeddings:")
    analyze_training_vs_test_separation(esm2_only_embeddings, training_genes, test_genes)
    
    if go_embeddings:
        print("\nGO embeddings:")
        analyze_training_vs_test_separation(go_only_embeddings, training_genes, test_genes)
    
    # 5. Coverage analysis
    print("\n5️⃣ Coverage analysis...")
    
    training_coverage = sum(1 for gene in training_genes if gene in esm2_embeddings)
    test_coverage = sum(1 for gene in test_genes if gene in esm2_embeddings)
    
    print(f"\n📊 COVERAGE SUMMARY:")
    print(f"   ESM2 coverage - Training: {training_coverage}/{len(training_genes)} ({100*training_coverage/len(training_genes):.1f}%)")
    print(f"   ESM2 coverage - Test: {test_coverage}/{len(test_genes)} ({100*test_coverage/len(test_genes):.1f}%)")
    
    if go_embeddings:
        training_go_coverage = sum(1 for gene in training_genes if gene in go_embeddings)
        test_go_coverage = sum(1 for gene in test_genes if gene in go_embeddings)
        print(f"   GO coverage - Training: {training_go_coverage}/{len(training_genes)} ({100*training_go_coverage/len(training_genes):.1f}%)")
        print(f"   GO coverage - Test: {test_go_coverage}/{len(test_genes)} ({100*test_go_coverage/len(test_genes):.1f}%)")
    
    # 6. Save summary report
    report_path = os.path.join(OUTPUT_DIR, "embedding_analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("ESM2 + GO Embedding Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Training genes: {len(training_genes)}\n")
        f.write(f"Test genes: {len(test_genes)}\n")
        f.write(f"Gene overlap: {len(training_genes.intersection(test_genes))}\n\n")
        
        f.write(f"ESM2 embeddings: {len(esm2_embeddings)} genes, {esm2_dim}D\n")
        f.write(f"GO embeddings: {len(go_embeddings) if go_embeddings else 0} genes, {go_dim}D\n")
        f.write(f"Combined embeddings: {len(combined_embeddings)} genes, {esm2_dim + go_dim}D\n\n")
        
        f.write(f"ESM2 coverage - Training: {100*training_coverage/len(training_genes):.1f}%\n")
        f.write(f"ESM2 coverage - Test: {100*test_coverage/len(test_genes):.1f}%\n")
        
        if go_embeddings:
            f.write(f"GO coverage - Training: {100*training_go_coverage/len(training_genes):.1f}%\n")
            f.write(f"GO coverage - Test: {100*test_go_coverage/len(test_genes):.1f}%\n")
    
    print(f"\n💾 Saved analysis report: {report_path}")
    
    print(f"\n🎉 EMBEDDING VISUALIZATION COMPLETE!")
    print(f"📊 All plots saved to: {OUTPUT_DIR}")
    print(f"📄 Analysis report: {report_path}")

if __name__ == "__main__":
    main()
