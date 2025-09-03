"""
STATE Model Implementation for Virtual Cell Challenge
Based on the original STATE architecture with adaptations for perturbation prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple, Any
import numpy as np
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, 
    LlamaRMSNorm,
    LlamaAttention,
    LlamaMLP
)


class NoRoPE(nn.Module):
    """No Rotary Position Embedding - used instead of standard RoPE"""
    def forward(self, x, seq_len=None):
        return x, x


class LlamaBidirectionalModel(nn.Module):
    """
    Bidirectional Llama model based on the transformer backbone from STATE
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = NoRoPE()

    def forward(self, inputs_embeds, attention_mask=None):
        hidden_states = inputs_embeds
        
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
        
        hidden_states = self.norm(hidden_states)
        return hidden_states


class SamplesLoss(nn.Module):
    """
    Custom loss function for STATE model
    """
    def __init__(self, alpha: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute MSE loss
        mse_loss = F.mse_loss(pred, target, reduction=self.reduction)
        
        # Compute Pearson correlation loss (simplified version)
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        target_centered = target - target.mean(dim=-1, keepdim=True)
        
        pred_std = pred_centered.std(dim=-1, keepdim=True) + 1e-8
        target_std = target_centered.std(dim=-1, keepdim=True) + 1e-8
        
        correlation = (pred_centered * target_centered).mean(dim=-1) / (pred_std * target_std).squeeze()
        correlation_loss = 1 - correlation.mean()
        
        # Combine losses
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * correlation_loss
        return total_loss


class PertSetsPerturbationModel(pl.LightningModule):
    """
    Main STATE model for perturbation prediction
    """
    
    def __init__(
        self,
        n_genes: int = 18080,
        n_perturbations: int = 19792,
        pert_emb_dim: int = 5120,  # ESM2 embedding dimension
        hidden_dim: int = 672,
        n_layers: int = 4,
        n_heads: int = 8,
        vocab_size: int = 32000,
        cell_sentence_len: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.n_genes = n_genes
        self.n_perturbations = n_perturbations
        self.pert_emb_dim = pert_emb_dim
        self.hidden_dim = hidden_dim
        self.cell_sentence_len = cell_sentence_len
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss function
        self.loss_fn = SamplesLoss()
        
        # Perturbation encoder (ESM2 features -> hidden_dim)
        self.pert_encoder = nn.Sequential(
            nn.Linear(pert_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Basal encoder (simplified to linear layer as mentioned in notebook)
        self.basal_encoder = nn.Linear(n_genes, hidden_dim)
        
        # Transformer backbone
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            intermediate_size=hidden_dim * 4,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            max_position_embeddings=cell_sentence_len,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
        )
        self.transformer_backbone = LlamaBidirectionalModel(config)
        
        # Output projection
        self.project_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes)
        )
        
        # Final compression and expansion layers
        self.final_down_then_up = nn.Sequential(
            nn.Linear(n_genes, n_genes // 8),  # Compression
            nn.GELU(),
            nn.Linear(n_genes // 8, n_genes)  # Expansion
        )
        
        self.relu = nn.ReLU()
        
    def forward(
        self, 
        gene_expression: torch.Tensor,
        perturbation_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the STATE model
        
        Args:
            gene_expression: Tensor of shape (batch_size, n_genes)
            perturbation_embeddings: Tensor of shape (batch_size, pert_emb_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Predicted gene expression: Tensor of shape (batch_size, n_genes)
        """
        batch_size = gene_expression.size(0)
        
        # Encode perturbation
        pert_encoded = self.pert_encoder(perturbation_embeddings)  # (batch_size, hidden_dim)
        
        # Encode basal gene expression
        basal_encoded = self.basal_encoder(gene_expression)  # (batch_size, hidden_dim)
        
        # Combine perturbation and basal encodings
        # Create sequence: [pert_encoded, basal_encoded, ...]
        # Pad to cell_sentence_len if needed
        combined = torch.stack([pert_encoded, basal_encoded], dim=1)  # (batch_size, 2, hidden_dim)
        
        # Pad to cell_sentence_len
        if combined.size(1) < self.cell_sentence_len:
            padding = torch.zeros(
                batch_size, 
                self.cell_sentence_len - combined.size(1), 
                self.hidden_dim,
                device=combined.device,
                dtype=combined.dtype
            )
            combined = torch.cat([combined, padding], dim=1)
        
        # Pass through transformer
        transformer_out = self.transformer_backbone(combined, attention_mask)  # (batch_size, seq_len, hidden_dim)
        
        # Take the mean of the sequence (or you could use the first token)
        pooled = transformer_out.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Project to gene space
        gene_pred = self.project_out(pooled)  # (batch_size, n_genes)
        
        # Apply final compression/expansion and residual connection
        residual = self.final_down_then_up(gene_pred)
        
        # Add residual connection with original gene expression
        output = gene_expression + residual
        
        return output
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        gene_expr = batch['gene_expression']
        pert_emb = batch['perturbation_embeddings']
        target = batch['target_expression']
        
        pred = self.forward(gene_expr, pert_emb)
        loss = self.loss_fn(pred, target)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        gene_expr = batch['gene_expression']
        pert_emb = batch['perturbation_embeddings']
        target = batch['target_expression']
        
        pred = self.forward(gene_expr, pert_emb)
        loss = self.loss_fn(pred, target)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1000,  # Adjust based on training steps
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step for inference"""
        gene_expr = batch['gene_expression']
        pert_emb = batch['perturbation_embeddings']
        
        return self.forward(gene_expr, pert_emb)


def create_state_model(config: Dict[str, Any]) -> PertSetsPerturbationModel:
    """
    Factory function to create STATE model with configuration
    """
    return PertSetsPerturbationModel(**config)