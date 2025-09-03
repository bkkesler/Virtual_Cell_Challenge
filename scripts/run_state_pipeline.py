"""
Example script to run the complete STATE model pipeline
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        print(f"Error: {e}")
        return False


def main():
    """Run the complete STATE model pipeline"""

    # Configuration
    config_path = "config/state_model_config.yaml"
    data_dir = "data/raw/single_cell_rnaseq/vcc_data"
    output_dir = "outputs/state_model_run"
    embeddings_path = f"{output_dir}/embeddings/esm2_embeddings.pt"
    gene_names_path = f"{data_dir}/gene_names.csv"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("🧬 Starting STATE Model Pipeline for Virtual Cell Challenge")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Config: {config_path}")

    # Get the absolute path to the training script (define it once here)
    project_root = Path.cwd()
    training_script = project_root / "scripts" / "training" / "train_state_model.py"

    # Check if the training script exists
    if not training_script.exists():
        print(f"❌ Training script not found at: {training_script}")
        print("Available files in scripts/:")
        scripts_dir = project_root / "scripts"
        if scripts_dir.exists():
            for item in scripts_dir.iterdir():
                print(f"  - {item.name}")
        return 1

    # Step 1: Create ESM2 embeddings (if needed)
    print("\n📊 Step 1: Creating ESM2 embeddings...")
    if not Path(embeddings_path).exists():
        cmd_embeddings = [
            sys.executable, str(training_script),
            "--config", config_path,
            "--data-dir", data_dir,
            "--output-dir", output_dir,
            "--create-embeddings",
            "--embeddings-path", embeddings_path,
            "--gene-names-path", gene_names_path,
            "--max-steps", "1"  # Just create embeddings, don't train
        ]

        if not run_command(cmd_embeddings, "ESM2 embeddings creation"):
            print("❌ Failed to create embeddings. Exiting.")
            return
    else:
        print(f"✅ Embeddings already exist at {embeddings_path}")

    # Step 2: Train the model
    print("\n🏋️ Step 2: Training STATE model...")
    cmd_train = [
        sys.executable, str(training_script),
        "--config", config_path,
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--embeddings-path", embeddings_path,
        "--gene-names-path", gene_names_path,
        "--max-steps", "400",
        "--batch-size", "32",
        "--learning-rate", "1e-4",
        "--gpus", "1",
        "--precision", "16-mixed"
    ]

    if not run_command(cmd_train, "Model training"):
        print("❌ Training failed. Exiting.")
        return

    # Step 3: Find the best checkpoint
    print("\n🔍 Step 3: Finding best checkpoint...")
    checkpoint_dir = Path(output_dir) / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if checkpoints:
            # Use the last checkpoint or find the best one
            best_checkpoint = checkpoint_dir / "last.ckpt"
            if not best_checkpoint.exists():
                best_checkpoint = sorted(checkpoints)[-1]  # Use the latest one
            print(f"✅ Using checkpoint: {best_checkpoint}")
        else:
            print("❌ No checkpoints found!")
            return
    else:
        print("❌ Checkpoint directory not found!")
        return

    # Step 4: Run inference on validation data
    print("\n🔮 Step 4: Running inference...")
    validation_template = f"{data_dir}/competition_val_template.h5ad"
    if not Path(validation_template).exists():
        print(f"⚠️  Validation template not found at {validation_template}")
        print("Creating a dummy validation file from training data...")
        # You might need to create this from your training data
        validation_template = f"{data_dir}/adata_Training.h5ad"

    predictions_path = f"{output_dir}/predictions.h5ad"
    config_file = f"{output_dir}/config.yaml"

    # Get the inference script path
    project_root = Path.cwd()
    inference_script = project_root / "scripts" / "inference" / "infer_state_model.py"

    # Check if the inference script exists
    if not inference_script.exists():
        print(f"❌ Inference script not found at: {inference_script}")
        return 1

    cmd_infer = [
        sys.executable, str(inference_script),
        "--checkpoint", str(best_checkpoint),
        "--config", config_file,
        "--input", validation_template,
        "--embeddings", embeddings_path,
        "--output", predictions_path,
        "--pert-col", "target_gene",
        "--batch-size", "128",
        "--device", "cuda"
    ]

    if not run_command(cmd_infer, "Model inference"):
        print("❌ Inference failed. Exiting.")
        return

    # Step 5: Prepare submission (optional)
    print("\n📦 Step 5: Preparing submission file...")
    cmd_submission = [
        sys.executable, str(inference_script),
        "--checkpoint", str(best_checkpoint),
        "--config", config_file,
        "--input", validation_template,
        "--embeddings", embeddings_path,
        "--output", predictions_path,
        "--prepare-submission",
        "--gene-names", gene_names_path
    ]

    # This step might fail if cell-eval is not installed, so we'll continue regardless
    submission_success = run_command(cmd_submission, "Submission preparation")

    if not submission_success:
        print("⚠️  Submission preparation failed. You may need to install cell-eval:")
        print("pip install git+https://github.com/ArcInstitute/cell-eval@main")

    # Summary
    print("\n" + "="*60)
    print("🎉 STATE Model Pipeline Summary")
    print("="*60)
    print(f"✅ Model trained and saved to: {output_dir}")
    print(f"✅ Best checkpoint: {best_checkpoint}")
    print(f"✅ Predictions saved to: {predictions_path}")
    if submission_success:
        print(f"✅ Submission file prepared")
    print("\n📊 Next steps:")
    print("1. Evaluate your predictions using the Virtual Cell Challenge metrics")
    print("2. Submit your .vcc file to the leaderboard")
    print("3. Iterate on model architecture and hyperparameters")
    print("\n🚀 Happy modeling!")


if __name__ == "__main__":
    main()