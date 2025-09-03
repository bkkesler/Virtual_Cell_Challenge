"""
Setup script to create the correct directory structure for STATE model
"""

import os
import shutil
from pathlib import Path


def main():
    """Create the necessary directories and move files"""

    print("🔧 Setting up STATE model directory structure...")

    # Create necessary directories
    directories = [
        "scripts/training",
        "scripts/inference",
        "src/models/state_model",
        "src/data_processing/loaders",
        "config",
        "outputs/state_model_run/embeddings",
        "outputs/state_model_run/checkpoints"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Create __init__.py files for Python packages
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/state_model/__init__.py",
        "src/data_processing/__init__.py",
        "src/data_processing/loaders/__init__.py"
    ]

    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created: {init_file}")

    # Check if the pipeline script exists in the wrong location
    old_pipeline = Path("scripts/state_model_pipeline.py")
    new_pipeline = Path("scripts/run_state_pipeline.py")

    if old_pipeline.exists() and not new_pipeline.exists():
        shutil.move(str(old_pipeline), str(new_pipeline))
        print(f"✅ Moved {old_pipeline} to {new_pipeline}")

    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Make sure you have all the required dependencies installed")
    print("2. Copy the model files to their correct locations (I'll provide them)")
    print("3. Run the pipeline: python scripts/run_state_pipeline.py")


if __name__ == "__main__":
    main()