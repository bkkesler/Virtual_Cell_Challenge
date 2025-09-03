import os
from pathlib import Path


def create_virtual_cell_project_structure(base_path="D:/Virtual_Cell3"):
    """
    Creates a clean folder structure for the Virtual Cell Challenge project.
    Just the directories and basic files - no complex implementations.
    """

    # Define the folder structure
    folders = [
        # Main project structure
        "",

        # Source code organization
        "src",
        "src/data_processing",
        "src/data_processing/loaders",
        "src/data_processing/preprocessors",
        "src/embeddings",
        "src/embeddings/go_embeddings",
        "src/embeddings/gene_embeddings",
        "src/embeddings/cell_embeddings",
        "src/models",
        "src/models/state_model",
        "src/models/random_forest",
        "src/models/neural_networks",
        "src/models/collaborative_filtering",
        "src/models/graph_neural_networks",
        "src/evaluation",
        "src/utils",

        # Notebooks for exploration and visualization
        "analysis",
        "analysis/exploratory_data_analysis",
        "analysis/embedding_visualization",
        "analysis/model_comparison",
        "analysis/results_analysis",

        # Configuration files
        "config",

        # Scripts for running pipelines
        "scripts",
        "scripts/preprocessing",
        "scripts/training",
        "scripts/inference",

        # Data directories (large files, not for git)
        "data",
        "data/raw",
        "data/raw/single_cell_rnaseq",
        "data/raw/go_terms",
        "data/raw/gene_annotations",
        "data/processed",
        "data/processed/embeddings",
        "data/processed/features",
        "data/interim",

        # Model artifacts (large files, not for git)
        "models",
        "models/embeddings",
        "models/state_model",
        "models/random_forest",
        "models/neural_networks",
        "models/collaborative_filtering",
        "models/graph_neural_networks",
        "models/ensemble",

        # Outputs (large files, not for git)
        "outputs",
        "outputs/predictions",
        "outputs/h5ad_files",
        "outputs/vcc_files",
        "outputs/evaluation_results",
        "outputs/figures",
        "outputs/plots",
        "outputs/visualizations",

        # Logs and temporary files
        "logs",
        "temp",

        # Documentation
        "docs",
        "docs/model_documentation",
        "docs/data_documentation",

        # Tests
        "tests",
        "tests/unit",
        "tests/integration",
    ]

    # Create base directory
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create all folders
    for folder in folders:
        if folder:  # Skip empty string (base directory)
            folder_path = base_dir / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {folder_path}")

    # Create basic files
    files_to_create = {
        # Root level files
        "README.md": create_readme(),
        "requirements.txt": create_requirements(),
        ".gitignore": create_gitignore(),

        # Config files
        "config/model_config.yaml": create_model_config(),
        "config/data_config.yaml": create_data_config(),

        # Basic __init__.py files
        "src/__init__.py": "",
        "src/data_processing/__init__.py": "",
        "src/data_processing/loaders/__init__.py": "",
        "src/data_processing/preprocessors/__init__.py": "",
        "src/embeddings/__init__.py": "",
        "src/models/__init__.py": "",
        "src/models/state_model/__init__.py": "",
        "src/evaluation/__init__.py": "",
        "src/utils/__init__.py": "",
        "tests/__init__.py": "",

        # Placeholder files to keep directories in git
        "data/raw/single_cell_rnaseq/.gitkeep": "",
        "data/raw/go_terms/.gitkeep": "",
        "data/raw/gene_annotations/.gitkeep": "",
        "data/processed/.gitkeep": "",
        "models/.gitkeep": "",
        "outputs/.gitkeep": "",
        "logs/.gitkeep": "",
        "temp/.gitkeep": "",

        # Basic documentation
        "docs/PROJECT_STRUCTURE.md": create_structure_doc(),
    }

    # Create all files
    for file_path, content in files_to_create.items():
        full_path = base_dir / file_path
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created file: {full_path}")

    print(f"\n✅ Virtual Cell Challenge project structure created at: {base_path}")
    print("\n📁 Key directories:")
    print("   - src/: Source code organized by functionality")
    print("   - data/: Large datasets (excluded from git)")
    print("   - models/: Trained model artifacts (excluded from git)")
    print("   - outputs/: Results and predictions (excluded from git)")
    print("   - analysis/: Python analysis scripts for exploration")
    print("   - scripts/: Executable Python scripts")
    print("   - config/: Configuration files")
    print("\n🎯 Next steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Download data to data/raw/single_cell_rnaseq/")
    print("   3. Start implementing your models in src/models/")
    print("   4. Use analysis/ scripts for exploration and visualization")
    print("   5. Save plots to outputs/figures/, outputs/plots/, outputs/visualizations/")


def create_readme():
    return """# Virtual Cell Challenge Project

This project implements multiple machine learning approaches for the Virtual Cell Challenge.

## Project Structure

- `src/`: Source code organized by functionality
- `data/`: Large datasets (not tracked in git)
- `models/`: Trained model artifacts (not tracked in git)
- `outputs/`: Results and predictions (not tracked in git)
- `analysis/`: Python analysis scripts for exploration and visualization
- `scripts/`: Executable Python scripts
- `config/`: Configuration files

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Configure data paths in `config/data_config.yaml`
3. Download competition data to `data/raw/single_cell_rnaseq/`
4. Start implementing models in `src/models/`
5. Use `analysis/` scripts for exploration and save plots to `outputs/`

## Analysis and Visualization

Since this project uses PyCharm (non-Jupyter edition), all analysis is done through Python scripts:
- Run analysis scripts from `analysis/` folder
- Visualizations automatically saved to `outputs/figures/`, `outputs/plots/`, `outputs/visualizations/`
- High-quality PNG files with 300 DPI for publications

## Models to Implement

- STATE Model (Arc Institute's context generalization model)
- Random Forest
- Neural Networks
- Collaborative Filtering
- Graph Neural Networks
- Ensemble methods

## Data

- Virtual Cell Challenge competition data
- Replogle support datasets for multi-context training
- GO term annotations
- Gene embeddings and features

## Outputs

- H5AD files with predicted expression
- VCC files for competition submission
- Evaluation metrics and visualizations
"""


def create_requirements():
    return """# Core ML libraries
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0

# Deep learning
torch>=1.11.0
torch-geometric>=2.0.0
pytorch-lightning>=1.9.0
transformers>=4.20.0

# Single-cell analysis
scanpy>=1.8.0
anndata>=0.8.0
h5py>=3.6.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Utilities
pyyaml>=6.0
tqdm>=4.62.0
joblib>=1.1.0
requests>=2.28.0

# Jupyter
# jupyter>=1.0.0  # Commented out since PyCharm edition doesn't support notebooks
# ipywidgets>=7.6.0  # Not needed without Jupyter

# Logging and experiment tracking
wandb>=0.13.0

# Testing
pytest>=6.2.0
pytest-cov>=3.0.0

# Development
black>=22.0.0
flake8>=4.0.0

# Additional dependencies for your existing scripts
gseapy>=1.0.0  # For GO analysis in Random Forest script

# Note: pickle, gc, os, sys, json, warnings are built-in Python modules
"""


def create_gitignore():
    return """# Large data files and model outputs
data/
!data/*/.gitkeep
models/
!models/.gitkeep
outputs/
!outputs/.gitkeep
temp/
!temp/.gitkeep
logs/
!logs/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# Virtual environments
venv/
env/
ENV/
.venv/

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp

# Large files
*.h5ad
*.vcc
*.h5
*.hdf5
*.pt

# Wandb
wandb/
"""


def create_model_config():
    return """# Model Configuration

state_model:
  n_genes: 18080
  pert_embed_dim: 5120  # ESM2 embedding dimension
  hidden_dim: 672
  num_layers: 4
  dropout: 0.1

random_forest:
  n_estimators: 100
  max_depth: 10
  random_state: 42
  n_jobs: -1

neural_network:
  hidden_layers: [512, 256, 128]
  dropout: 0.2
  learning_rate: 0.001
  batch_size: 64
  epochs: 100

collaborative_filtering:
  n_factors: 50
  learning_rate: 0.01
  regularization: 0.1
  epochs: 100

graph_neural_network:
  hidden_dim: 64
  num_layers: 3
  dropout: 0.1
  learning_rate: 0.001
  batch_size: 32

ensemble:
  method: "weighted_average"
  weights: [0.4, 0.2, 0.2, 0.1, 0.1]  # STATE, RF, NN, CF, GNN
"""


def create_data_config():
    return """# Data Configuration

paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  single_cell_data: "data/raw/single_cell_rnaseq"
  go_terms: "data/raw/go_terms"
  gene_annotations: "data/raw/gene_annotations"

  # Output paths
  predictions: "outputs/predictions"
  h5ad_output: "outputs/h5ad_files"
  vcc_output: "outputs/vcc_files"

# Virtual Cell Challenge datasets
competition_datasets:
  train: "data/raw/single_cell_rnaseq/competition_train.h5ad"
  val: "data/raw/single_cell_rnaseq/competition_val_template.h5ad"

# Replogle support datasets for STATE training  
replogle_datasets:
  k562_gwps: "data/raw/single_cell_rnaseq/k562_gwps.h5"
  rpe1: "data/raw/single_cell_rnaseq/rpe1.h5"
  jurkat: "data/raw/single_cell_rnaseq/jurkat.h5"
  k562: "data/raw/single_cell_rnaseq/k562.h5"
  hepg2: "data/raw/single_cell_rnaseq/hepg2.h5"

preprocessing:
  min_cells: 3
  min_genes: 200
  max_genes: 5000
  mt_threshold: 20
  normalize: true
  log_transform: true
  scale: false
  target_sum: 10000

embeddings:
  gene_embedding_dim: 128
  cell_embedding_dim: 64
  go_embedding_dim: 50
"""


def create_structure_doc():
    return """# Project Structure Documentation

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
"""


if __name__ == "__main__":
    create_virtual_cell_project_structure()