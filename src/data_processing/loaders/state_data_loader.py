"""
Data loader for STATE model training and inference
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import anndata as ad
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import h5py
from pathlib import Path


class VirtualCellDataset(Dataset):
    """
    Dataset class for Virtual Cell Challenge data
    """
    
    def __init__(
        self,
        adata: ad.AnnData,
        perturbation_embeddings: Dict[str, np.ndarray],
        pert_col: str = 'target_gene',
        batch_col: str = 'batch_var',
        control_name: str = 'non-targeting',
        transform_type: str = 'log1p'
    ):
        """
        Initialize dataset
        
        Args:
            adata: AnnData object with gene expression data
            perturbation_embeddings: Dictionary mapping perturbation names to embeddings
            pert_col: Column name for perturbation information
            batch_col: Column name for batch information
            control_name: Name of control perturbation
            transform_type: Type of transformation to apply
        """
        self.adata = adata
        self.perturbation_embeddings = perturbation_embeddings
        self.pert_col = pert_col
        self.batch_col = batch_col
        self.control_name = control_name
        self.transform_type = transform_type
        
        # Get gene expression matrix
        if hasattr(adata.X, 'toarray'):
            self.gene_expression = adata.X.toarray()
        else:
            self.gene_expression = adata.X
            
        # Apply transformation
        if transform_type == 'log1p':
            self.gene_expression = np.log1p(self.gene_expression)
        
        # Get perturbation information
        self.perturbations = adata.obs[pert_col].values
        self.batches = adata.obs[batch_col].values if batch_col in adata.obs.columns else None
        
        # Create perturbation to embedding mapping
        self.pert_to_embedding = {}
        for pert in np.unique(self.perturbations):
            if pert in perturbation_embeddings:
                self.pert_to_embedding[pert] = perturbation_embeddings[pert]
            else:
                # Use control embedding for unknown perturbations
                self.pert_to_embedding[pert] = perturbation_embeddings.get(
                    control_name, 
                    np.zeros(list(perturbation_embeddings.values())[0].shape)
                )
    
    def __len__(self) -> int:
        return len(self.adata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index
        
        Returns:
            Dictionary with gene expression, perturbation embedding, and target
        """
        gene_expr = torch.FloatTensor(self.gene_expression[idx])
        pert_name = self.perturbations[idx]
        pert_emb = torch.FloatTensor(self.pert_to_embedding[pert_name])
        
        # For training, target is the same as input (autoencoder-like setup)
        # In practice, you might want to use perturbed expression as target
        target_expr = gene_expr.clone()
        
        batch_info = torch.LongTensor([0]) if self.batches is None else torch.LongTensor([hash(self.batches[idx]) % 1000])
        
        return {
            'gene_expression': gene_expr,
            'perturbation_embeddings': pert_emb,
            'target_expression': target_expr,
            'perturbation_name': pert_name,
            'batch_info': batch_info
        }


class VirtualCellDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for Virtual Cell data
    """
    
    def __init__(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        perturbation_embeddings_path: str = None,
        perturbation_embeddings: Optional[Dict[str, np.ndarray]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pert_col: str = 'target_gene',
        batch_col: str = 'batch_var',
        control_name: str = 'non-targeting',
        **kwargs
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.perturbation_embeddings_path = perturbation_embeddings_path
        self.perturbation_embeddings = perturbation_embeddings
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pert_col = pert_col
        self.batch_col = batch_col
        self.control_name = control_name
        self.kwargs = kwargs
        
    def prepare_data(self):
        """Download and prepare data"""
        # Check if data files exist
        if not Path(self.train_data_path).exists():
            raise FileNotFoundError(f"Training data not found: {self.train_data_path}")
            
        if self.val_data_path and not Path(self.val_data_path).exists():
            raise FileNotFoundError(f"Validation data not found: {self.val_data_path}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        # Load perturbation embeddings
        if self.perturbation_embeddings is None:
            if self.perturbation_embeddings_path:
                self.perturbation_embeddings = self.load_perturbation_embeddings(
                    self.perturbation_embeddings_path
                )
            else:
                raise ValueError("Must provide either perturbation_embeddings or perturbation_embeddings_path")
        
        # Load training data
        train_adata = ad.read_h5ad(self.train_data_path)
        self.train_dataset = VirtualCellDataset(
            train_adata,
            self.perturbation_embeddings,
            pert_col=self.pert_col,
            batch_col=self.batch_col,
            control_name=self.control_name
        )
        
        # Load validation data
        if self.val_data_path:
            val_adata = ad.read_h5ad(self.val_data_path)
            self.val_dataset = VirtualCellDataset(
                val_adata,
                self.perturbation_embeddings,
                pert_col=self.pert_col,
                batch_col=self.batch_col,
                control_name=self.control_name
            )
        else:
            # Split training data
            train_size = int(0.8 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size]
            )
    
    def train_dataloader(self) -> DataLoader:
        """Training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Prediction dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    @staticmethod
    def load_perturbation_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
        """
        Load perturbation embeddings from file
        
        Args:
            embeddings_path: Path to embeddings file (h5, npz, or pt)
            
        Returns:
            Dictionary mapping perturbation names to embeddings
        """
        embeddings_path = Path(embeddings_path)
        
        if embeddings_path.suffix == '.h5':
            with h5py.File(embeddings_path, 'r') as f:
                embeddings = {}
                for key in f.keys():
                    embeddings[key] = f[key][:]
                return embeddings
        
        elif embeddings_path.suffix == '.npz':
            data = np.load(embeddings_path)
            return {key: data[key] for key in data.keys()}
        
        elif embeddings_path.suffix == '.pt':
            # Handle PyTorch 2.6+ weights_only security feature
            try:
                return torch.load(embeddings_path, weights_only=False)
            except Exception as e:
                if "weights_only" in str(e):
                    print(f"⚠️  PyTorch security warning: {e}")
                    print("Loading embeddings with weights_only=False (trusted source)")
                    return torch.load(embeddings_path, weights_only=False)
                else:
                    raise e
        
        else:
            raise ValueError(f"Unsupported file format: {embeddings_path.suffix}")


def create_esm2_embeddings(gene_names: List[str], model_name: str = "facebook/esm2_t33_650M_UR50D") -> Dict[str, np.ndarray]:
    """
    Create ESM2 embeddings for gene names
    
    Args:
        gene_names: List of gene names
        model_name: ESM2 model name
        
    Returns:
        Dictionary mapping gene names to embeddings
    """
    from transformers import EsmTokenizer, EsmModel
    
    # Load ESM2 model and tokenizer
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.eval()
    
    embeddings = {}
    
    with torch.no_grad():
        for gene_name in gene_names:
            # Tokenize gene name
            inputs = tokenizer(gene_name, return_tensors="pt", padding=True, truncation=True)
            
            # Get embeddings
            outputs = model(**inputs)
            
            # Use mean pooling of last hidden states
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings[gene_name] = embedding
    
    return embeddings