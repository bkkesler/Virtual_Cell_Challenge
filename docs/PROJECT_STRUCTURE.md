# Project Structure Documentation

## Overview
This project is organized to support multiple machine learning approaches for the Virtual Cell Challenge.

## Directory Structure

### `/src/` - Source Code
- `data_processing/`: Data loading and preprocessing
  - `loaders/`: Dataset loading utilities
  - `preprocessors/`: Data preprocessing pipelines
- `embeddings/`: Embedding generation modules
- `models/`: Model implementations
  - `state_model/`: STATE model implementation
  - `random_forest/`: Random Forest models
  - `neural_networks/`: Deep learning models
  - `collaborative_filtering/`: Collaborative filtering
  - `graph_neural_networks/`: GNN implementations
- `evaluation/`: Model evaluation utilities
- `utils/`: General utility functions

### `/data/` - Data Storage (Not in Git)
- `raw/`: Original datasets
  - `single_cell_rnaseq/`: Competition and support datasets
  - `go_terms/`: GO term annotations
  - `gene_annotations/`: Gene metadata
- `processed/`: Preprocessed data
- `interim/`: Intermediate processing results

### `/models/` - Model Artifacts (Not in Git)
- Trained model checkpoints organized by approach

### `/outputs/` - Results (Not in Git)
- `predictions/`: Model predictions
- `h5ad_files/`: AnnData format outputs
- `vcc_files/`: Competition submission files
- `evaluation_results/`: Performance metrics
- `figures/`: Generated visualizations

### `/analysis/` - Analysis Scripts
- Python scripts for data exploration and visualization
- Automatically save plots to outputs/ directories
- Replace Jupyter notebooks for PyCharm compatibility

### `/scripts/` - Executable Scripts
- Training pipelines
- Inference scripts
- Data preprocessing utilities

### `/config/` - Configuration Files
- Model hyperparameters
- Data paths and settings

## Getting Started

1. Install dependencies from `requirements.txt`
2. Download competition data to `data/raw/single_cell_rnaseq/`
3. Configure paths in `config/data_config.yaml`
4. Start implementing models in `src/models/`
5. Use notebooks for exploration and analysis

## Best Practices

- Keep large files out of git (data/, models/, outputs/)
- Use configuration files for parameters
- Implement proper logging
- Write tests for critical functions
- Document your implementations
