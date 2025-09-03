# STATE Model Implementation for Virtual Cell Challenge

This implementation provides a PyTorch Lightning-based STATE model for the Virtual Cell Challenge, adapted from the original STATE architecture for context generalization in single-cell perturbation prediction.

## 🏗️ Architecture Overview

The STATE model consists of:

1. **Perturbation Encoder**: Transforms ESM2 gene embeddings (5120D) to hidden space (672D)
2. **Basal Encoder**: Linear projection of gene expression to hidden space
3. **Transformer Backbone**: 4-layer bidirectional Llama-based transformer
4. **Output Projection**: Projects hidden representations back to gene expression space
5. **Residual Connection**: Adds predicted changes to original expression

## 📦 Installation

### 1. Install Additional Requirements

Update your `requirements.txt` with the new dependencies and install:

```bash
pip install einops torchmetrics hydra-core omegaconf lightning
pip install transformers torch pytorch-lightning
```

### 2. Install cell-eval (for submission preparation)

```bash
pip install git+https://github.com/ArcInstitute/cell-eval@main
```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline

Use the automated pipeline script:

```bash
python scripts/run_state_pipeline.py
```

This will:
1. Create ESM2 embeddings for gene names
2. Train the STATE model
3. Run inference on validation data
4. Prepare submission file

### Option 2: Step-by-Step Execution

#### Step 1: Create ESM2 Embeddings

```bash
python scripts/training/train_state_model.py \
    --config config/state_model_config.yaml \
    --data-dir data/raw/single_cell_rnaseq/vcc_data \
    --output-dir outputs/state_model \
    --create-embeddings \
    --embeddings-path outputs/state_model/embeddings/esm2_embeddings.pt \
    --gene-names-path data/raw/single_cell_rnaseq/vcc_data/gene_names.csv \
    --max-steps 1
```

#### Step 2: Train the Model

```bash
python scripts/training/train_state_model.py \
    --config config/state_model_config.yaml \
    --data-dir data/raw/single_cell_rnaseq/vcc_data \
    --output-dir outputs/state_model \
    --embeddings-path outputs/state_model/embeddings/esm2_embeddings.pt \
    --max-steps 400 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --gpus 1 \
    --log-wandb
```

#### Step 3: Run Inference

```bash
python scripts/inference/infer_state_model.py \
    --checkpoint outputs/state_model/checkpoints/last.ckpt \
    --config outputs/state_model/config.yaml \
    --input data/raw/single_cell_rnaseq/vcc_data/validation_template.h5ad \
    --embeddings outputs/state_model/embeddings/esm2_embeddings.pt \
    --output outputs/state_model/predictions.h5ad \
    --prepare-submission \
    --gene-names data/raw/single_cell_rnaseq/vcc_data/gene_names.csv
```

## ⚙️ Configuration

The model configuration is in `config/state_model_config.yaml`. Key parameters:

### Model Architecture
- `hidden_dim`: Transformer hidden dimension (default: 672)
- `n_layers`: Number of transformer layers (default: 4)
- `n_heads`: Number of attention heads (default: 8)
- `dropout`: Dropout rate (default: 0.1)

### Training
- `learning_rate`: Learning rate (default: 1e-4)
- `batch_size`: Batch size (default: 32)
- `max_steps`: Maximum training steps (default: 400)

### Data
- `pert_col`: Column name for perturbation info (default: "target_gene")
- `batch_col`: Column name for batch info (default: "batch_var")

## 📊 Data Requirements

Your data directory should contain:

```
data/raw/single_cell_rnaseq/vcc_data/
├── adata_Training.h5ad          # Training data
├── gene_names.csv               # Gene names for ESM2 embeddings
├── validation_template.h5ad     # Validation template (optional)
└── pert_counts_Validation.csv   # Perturbation counts (optional)
```

## 🧠 Model Components

### Core Files
- `src/models/state_model/state_model.py`: Main model implementation
- `src/data_processing/loaders/state_data_loader.py`: Data loading and preprocessing
- `scripts/training/train_state_model.py`: Training script
- `scripts/inference/infer_state_model.py`: Inference script

### Key Classes
- `PertSetsPerturbationModel`: Main STATE model (PyTorch Lightning module)
- `VirtualCellDataModule`: Data module for loading and preprocessing
- `VirtualCellDataset`: Dataset class for single-cell data
- `SamplesLoss`: Custom loss function combining MSE and correlation

## 🔧 Customization

### Modify Model Architecture

Edit `config/state_model_config.yaml` or override in training script:

```bash
python scripts/training/train_state_model.py \
    --config config/state_model_config.yaml \
    --batch-size 64 \
    --learning-rate 5e-5 \
    --max-steps 1000
```

### Use Different Embeddings

The model uses ESM2 embeddings by default. To use custom embeddings:

1. Create embeddings dictionary: `{gene_name: embedding_vector}`
2. Save as PyTorch file: `torch.save(embeddings, 'custom_embeddings.pt')`
3. Pass to training: `--embeddings-path custom_embeddings.pt`

### Add New Loss Functions

Modify `SamplesLoss` in `src/models/state_model/state_model.py`:

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # Your custom loss implementation
        return loss
```

## 📈 Monitoring Training

### Weights & Biases
```bash
python scripts/training/train_state_model.py \
    --config config/state_model_config.yaml \
    --log-wandb
```

### Local Logs
Training logs are saved to