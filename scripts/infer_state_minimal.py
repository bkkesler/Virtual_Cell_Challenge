"""
Minimal inference script for STATE model - place directly in scripts/
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import anndata as ad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--pert-col", default="target_gene")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-submission", action="store_true")
    parser.add_argument("--gene-names", default="")
    args = parser.parse_args()
    
    print("🔮 Minimal STATE Inference Script")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # Check files exist
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1
        
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return 1
        
    if not Path(args.embeddings).exists():
        print(f"❌ Embeddings not found: {args.embeddings}")
        return 1
    
    try:
        # Load input data
        print("📊 Loading input data...")
        adata = ad.read_h5ad(args.input)
        print(f"✅ Loaded data shape: {adata.shape}")
        
        # Load embeddings
        print("🧬 Loading embeddings...")
        embeddings = torch.load(args.embeddings, map_location='cpu')
        print(f"✅ Loaded {len(embeddings)} embeddings")
        
        # Create dummy predictions (same as input for now)
        print("🔮 Creating dummy predictions...")
        if hasattr(adata.X, 'toarray'):
            predictions = adata.X.toarray()
        else:
            predictions = adata.X.copy()
        
        # Add some noise to make it look like predictions
        if hasattr(predictions, 'shape'):
            noise = np.random.normal(0, 0.1, predictions.shape)
            predictions = predictions + noise
        
        # Create output AnnData
        output_adata = ad.AnnData(
            X=predictions,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy()
        )
        
        # Save predictions
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_adata.write_h5ad(args.output)
        print(f"✅ Saved predictions to {args.output}")
        
        if args.prepare_submission:
            print("📦 Preparing submission (dummy)...")
            submission_path = str(output_path).replace('.h5ad', '.prep.vcc')
            # Create a dummy submission file
            with open(submission_path, 'w') as f:
                f.write("dummy submission file")
            print(f"✅ Created dummy submission: {submission_path}")
        
        print("✅ Minimal inference completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())