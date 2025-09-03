"""
Script to copy STATE model files from Downloads to Virtual_Cell3 project
Run this script from your D:\Virtual_Cell3 directory
"""

import os
import shutil
from pathlib import Path


def copy_file_safe(source_path, dest_path, description):
    """Safely copy a file with error handling"""
    try:
        # Create destination directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            print(f"✅ Copied {description}")
            print(f"   From: {source_path}")
            print(f"   To:   {dest_path}")
            return True
        else:
            print(f"⚠️  {description} not found at {source_path}")
            return False
    except Exception as e:
        print(f"❌ Error copying {description}: {e}")
        return False


def create_init_files():
    """Create __init__.py files for Python packages"""
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py", 
        "src/models/state_model/__init__.py",
        "src/data_processing/__init__.py",
        "src/data_processing/loaders/__init__.py",
        "src/evaluation/__init__.py",
        "src/utils/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = Path(init_file)
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.touch()
        print(f"✅ Created: {init_file}")


def main():
    """Main function to copy all STATE model files"""
    
    # Define paths
    downloads_dir = Path("D:/Downloads")
    project_root = Path("D:/Virtual_Cell3")
    
    print("🔧 STATE Model File Copy Script")
    print(f"Downloads directory: {downloads_dir}")
    print(f"Project root: {project_root}")
    print("="*60)
    
    # Change to project directory
    os.chdir(project_root)
    print(f"Working directory: {Path.cwd()}")
    
    # File mapping: (download_filename, destination_path, description)
    file_mappings = [
        # Core model files
        ("state_model.py", "src/models/state_model/state_model.py", "STATE Model Core"),
        ("state_data_loader.py", "src/data_processing/loaders/state_data_loader.py", "Data Loader"),
        
        # Scripts
        ("train_state_model.py", "scripts/training/train_state_model.py", "Training Script"),
        ("infer_state_model.py", "scripts/inference/infer_state_model.py", "Inference Script"),
        ("run_state_pipeline.py", "scripts/run_state_pipeline.py", "Main Pipeline Script"),
        
        # Minimal scripts for testing
        ("train_state_minimal.py", "scripts/train_state_minimal.py", "Minimal Training Script"),
        ("infer_state_minimal.py", "scripts/infer_state_minimal.py", "Minimal Inference Script"),
        ("run_minimal_pipeline.py", "scripts/run_minimal_pipeline.py", "Minimal Pipeline Script"),
        
        # Configuration
        ("state_model_config.yaml", "config/state_model_config.yaml", "Model Configuration"),
        
        # Documentation
        ("STATE_MODEL_README.md", "docs/STATE_MODEL_README.md", "Documentation"),
        
        # Setup and utilities
        ("setup_state_model.py", "setup_state_model.py", "Setup Script"),
        ("copy_state_files.py", "copy_state_files.py", "This Copy Script"),
        
        # Updated requirements
        ("requirements.txt", "requirements_state.txt", "Updated Requirements (backup)")
    ]
    
    print("\n📁 Creating directory structure...")
    
    # Create necessary directories
    directories = [
        "scripts/training",
        "scripts/inference", 
        "src/models/state_model",
        "src/data_processing/loaders",
        "src/evaluation",
        "src/utils",
        "config",
        "docs",
        "outputs/state_model_run/embeddings",
        "outputs/state_model_run/checkpoints",
        "outputs/state_model_test/embeddings",
        "outputs/state_model_test/checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")
    
    print("\n🔧 Creating __init__.py files...")
    create_init_files()
    
    print("\n📋 Copying files...")
    
    # Copy each file
    successful_copies = 0
    total_files = len(file_mappings)
    
    for download_name, dest_path, description in file_mappings:
        source_path = downloads_dir / download_name
        destination_path = Path(dest_path)
        
        if copy_file_safe(source_path, destination_path, description):
            successful_copies += 1
        print()  # Empty line for readability
    
    print("="*60)
    print(f"📊 Copy Summary: {successful_copies}/{total_files} files copied successfully")
    
    # Special handling for requirements.txt
    print("\n📦 Handling requirements.txt...")
    original_req = Path("requirements.txt")
    state_req = downloads_dir / "requirements.txt"
    
    if state_req.exists() and original_req.exists():
        # Backup original requirements
        shutil.copy2(original_req, "requirements_original_backup.txt")
        print("✅ Backed up original requirements.txt")
        
        # Copy new requirements
        shutil.copy2(state_req, original_req)
        print("✅ Updated requirements.txt with STATE model dependencies")
    elif state_req.exists():
        shutil.copy2(state_req, original_req)
        print("✅ Copied requirements.txt")
    
    print("\n🎯 Next Steps:")
    print("1. Install new dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Test the setup with minimal pipeline:")
    print("   python scripts/run_minimal_pipeline.py")
    print("\n3. If minimal pipeline works, run full pipeline:")
    print("   python scripts/run_state_pipeline.py")
    
    print("\n✨ File copy completed!")
    
    # Check if any important files are missing
    critical_files = [
        "src/models/state_model/state_model.py",
        "scripts/training/train_state_model.py",
        "config/state_model_config.yaml"
    ]
    
    missing_critical = []
    for critical_file in critical_files:
        if not Path(critical_file).exists():
            missing_critical.append(critical_file)
    
    if missing_critical:
        print("\n⚠️  Critical files missing:")
        for missing in missing_critical:
            print(f"   - {missing}")
        print("Please check that all files were downloaded correctly.")
    else:
        print("\n✅ All critical files are in place!")


if __name__ == "__main__":
    main()