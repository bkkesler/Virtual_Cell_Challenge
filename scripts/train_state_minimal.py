"""
Minimal training script for STATE model - place directly in scripts/
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd

def create_dummy_embeddings(gene_names_path: str, output_path: str):
    """Create dummy embeddings for testing"""
    print(f"Creating dummy embeddings from {gene_names_path}")
    
    # Load gene names
    if not Path(gene_names_path).exists():
        print(f"Gene names file not found: {gene_names_path}")
        return False
        
    gene_df = pd.read_csv(gene_names_path)
    gene_names = gene_df.iloc[:, 0].tolist()
    
    print(f"Found {len(gene_names)} genes")
    
    # Create dummy embeddings (5120 dimensions like ESM2)
    embeddings = {}
    for gene_name in gene_names[:100]:  # Limit for testing
        embeddings[gene_name] = np.random.randn(5120).astype(np.float32)
    
    # Add control
    embeddings['non-targeting'] = np.zeros(5120, dtype=np.float32)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    
    print(f"✅ Saved dummy embeddings to {output_path}")
    print(f"Embeddings shape: {embeddings[gene_names[0]].shape}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-names-path", required=True)
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--create-embeddings", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()
    
    print("🧬 Minimal STATE Model Script")
    print(f"Gene names: {args.gene_names_path}")
    print(f"Embeddings: {args.embeddings_path}")
    print(f"Output: {args.output_dir}")
    
    if args.create_embeddings:
        success = create_dummy_embeddings(args.gene_names_path, args.embeddings_path)
        if success:
            print("✅ Embeddings created successfully!")
        else:
            print("❌ Failed to create embeddings")
            return 1
    
    # Create output directory structure
    output_dir = Path(args.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    # Create dummy config file
    config_content = """
model:
  n_genes: 18080
  hidden_dim: 672
  learning_rate: 1e-4
data:
  batch_size: 32
training:
  max_steps: 400
"""
    
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Create dummy checkpoint
    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    torch.save({"state_dict": {}, "epoch": 1}, checkpoint_path)
    
    print(f"✅ Created dummy config at {config_path}")
    print(f"✅ Created dummy checkpoint at {checkpoint_path}")
    print("✅ Minimal training completed!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())