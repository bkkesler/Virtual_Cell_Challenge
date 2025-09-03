# Virtual Cell Challenge Project

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
