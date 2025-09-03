"""
Inference script for STATE model on Virtual Cell Challenge data
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import torch
import pytorch_lightning as pl
import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from models.state_model.state_model import PertSetsPerturbationModel
from data_processing.loaders.state_data_loader import VirtualCellDataModule


def load_model_from_checkpoint(checkpoint_path: str, config_path: str = None) -> PertSetsPerturbationModel:
    """
    Load STATE model from checkpoint

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file

    Returns:
        Loaded model
    """
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config['model']

        # Create model with config
        model = PertSetsPerturbationModel(
            n_genes=model_config.get('n_genes', 18080),
            n_perturbations=model_config.get('n_perturbations', 19792),
            pert_emb_dim=model_config.get('pert_emb_dim', 5120),
            hidden_dim=model_config.get('hidden_dim', 672),
            n_layers=model_config.get('n_layers', 4),
            n_heads=model_config.get('n_heads', 8),
            vocab_size=model_config.get('vocab_size', 32000),
            cell_sentence_len=model_config.get('cell_sentence_len', 128),
            dropout=model_config.get('dropout', 0.1),
            learning_rate=model_config.get('learning_rate', 1e-4),
            weight_decay=model_config.get('weight_decay', 0.01)
        )

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Load model directly from checkpoint
        model = PertSetsPerturbationModel.load_from_checkpoint(checkpoint_path)

    model.eval()
    return model


def run_inference(
        model: PertSetsPerturbationModel,
        input_adata_path: str,
        embeddings_path: str,
        output_path: str,
        pert_col: str = 'target_gene',
        batch_size: int = 128,
        device: str = 'cuda'
) -> None:
    """
    Run inference on input data

    Args:
        model: Trained STATE model
        input_adata_path: Path to input AnnData file
        embeddings_path: Path to perturbation embeddings
        output_path: Path to save predictions
        pert_col: Column name for perturbation information
        batch_size: Batch size for inference
        device: Device to run inference on
    """
    # Load input data
    print(f"Loading input data from {input_adata_path}")
    adata = ad.read_h5ad(input_adata_path)

    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = torch.load(embeddings_path, map_location='cpu')

    # Move model to device
    model = model.to(device)

    # Prepare data
    if hasattr(adata.X, 'toarray'):
        gene_expression = adata.X.toarray()
    else:
        gene_expression = adata.X

    # Apply log1p transformation if needed
    gene_expression = np.log1p(gene_expression)

    perturbations = adata.obs[pert_col].values

    # Create perturbation embeddings for each sample
    pert_embeddings = []
    for pert in perturbations:
        if pert in embeddings:
            pert_embeddings.append(embeddings[pert])
        else:
            # Use control embedding for unknown perturbations
            pert_embeddings.append(embeddings.get('non-targeting', np.zeros(5120)))

    pert_embeddings = np.array(pert_embeddings)

    # Run inference in batches
    predictions = []
    n_samples = len(adata)
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"Running inference on {n_samples} samples in {n_batches} batches...")

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            # Get batch data
            batch_gene_expr = torch.FloatTensor(gene_expression[start_idx:end_idx]).to(device)
            batch_pert_emb = torch.FloatTensor(pert_embeddings[start_idx:end_idx]).to(device)

            # Run inference
            batch_pred = model.forward(batch_gene_expr, batch_pert_emb)

            # Move to CPU and store
            predictions.append(batch_pred.cpu().numpy())

    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)

    # Create output AnnData
    output_adata = ad.AnnData(
        X=predictions,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        uns=adata.uns.copy()
    )

    # Save predictions
    print(f"Saving predictions to {output_path}")
    output_adata.write_h5ad(output_path)
    print("Inference completed!")


def main():
    parser = argparse.ArgumentParser(description="Run STATE model inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--input", type=str, required=True, help="Path to input AnnData file")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to perturbation embeddings")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--pert-col", type=str, default="target_gene", help="Perturbation column name")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--prepare-submission", action="store_true", help="Prepare submission file using cell-eval")
    parser.add_argument("--gene-names", type=str, help="Path to gene names CSV for submission preparation")
    args = parser.parse_args()

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Check if input file exists
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Check if embeddings exist
    if not Path(args.embeddings).exists():
        raise FileNotFoundError(f"Embeddings not found: {args.embeddings}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, args.config)

    # Run inference
    run_inference(
        model=model,
        input_adata_path=args.input,
        embeddings_path=args.embeddings,
        output_path=args.output,
        pert_col=args.pert_col,
        batch_size=args.batch_size,
        device=args.device
    )

    # Prepare submission if requested
    if args.prepare_submission:
        if not args.gene_names:
            raise ValueError("Gene names file required for submission preparation")

        print("Preparing submission file using cell-eval...")
        import subprocess

        # Prepare submission using cell-eval
        cmd = [
            "cell-eval", "prep",
            "-i", args.output,
            "-g", args.gene_names
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Submission preparation completed!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error preparing submission: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
        except FileNotFoundError:
            print("cell-eval not found. Please install it manually:")
            print("pip install git+https://github.com/ArcInstitute/cell-eval@main")


if __name__ == "__main__":
    main()