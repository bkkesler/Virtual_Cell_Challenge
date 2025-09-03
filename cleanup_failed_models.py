#!/usr/bin/env python3
"""
State Model Cleanup Script
Safely removes failed/incomplete state model runs while preserving successful ones
"""

import os
import shutil
import sys
from pathlib import Path
import glob

def get_folder_size(folder_path):
    """Calculate folder size in MB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except:
        return 0
    return total_size / (1024 * 1024)  # Convert to MB

def safe_remove_directory(path):
    """Safely remove a directory with error handling"""
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
    except Exception as e:
        print(f"   ❌ Error removing {path}: {e}")
        return False
    return False

def cleanup_state_models():
    """Main cleanup function"""
    
    print("🧹 STATE MODEL CLEANUP SCRIPT")
    print("=" * 50)
    print("This script will remove failed/incomplete state model runs")
    print("while preserving successful models and important files.\n")
    
    # Define the base paths
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print("❌ outputs/ directory not found!")
        return
    
    # =================================================================
    # ANALYSIS: Identify what to keep vs remove
    # =================================================================
    
    print("📊 ANALYZING STATE MODEL DIRECTORIES...")
    print("-" * 40)
    
    # Directories to KEEP (successful or important)
    keep_dirs = {
        "state_model_FINAL_SUCCESS": "✅ Final successful model with checkpoints - KEEP",
        "evaluation_results": "✅ Important evaluation files - KEEP",
        "vcc_submission": "✅ VCC submission files - KEEP",
        "figures": "✅ Analysis figures - KEEP",
        "plots": "✅ Plots and visualizations - KEEP",
        "predictions": "✅ Important predictions - KEEP",
        "visualizations": "✅ Visualization outputs - KEEP",
        "h5ad_files": "✅ Important data files - KEEP",
        "vcc_files": "✅ VCC related files - KEEP"
    }
    
    # Directories to REMOVE (failed attempts and logs-only)
    remove_dirs = {
        "state_model_run": "❌ Incomplete run - REMOVE",
        "state_model_run_complete": "❌ Incomplete despite name - REMOVE", 
        "state_model_run_fixed": "❌ Failed fix attempt - REMOVE",
        "state_model_run_small": "❌ Failed small run - REMOVE",
        "state_model_run_subset": "❌ Failed subset run - REMOVE",
        "state_model_run_SUCCESS": "❌ Logs only, no model files - REMOVE",
        "state_model_test": "❌ Test run - REMOVE"
    }
    
    total_space_to_free = 0
    dirs_to_remove = []
    
    # Scan outputs directory
    for item in outputs_dir.iterdir():
        if item.is_dir():
            dir_name = item.name
            size_mb = get_folder_size(item)
            
            if dir_name in keep_dirs:
                print(f"   {keep_dirs[dir_name]} | {size_mb:.1f} MB | {dir_name}")
            elif dir_name in remove_dirs:
                print(f"   {remove_dirs[dir_name]} | {size_mb:.1f} MB | {dir_name}")
                dirs_to_remove.append(item)
                total_space_to_free += size_mb
            else:
                print(f"   ❓ Unknown directory - REVIEW | {size_mb:.1f} MB | {dir_name}")
    
    print(f"\n💾 SPACE ANALYSIS:")
    print(f"   Total space to free: {total_space_to_free:.1f} MB ({total_space_to_free/1024:.2f} GB)")
    print(f"   Directories to remove: {len(dirs_to_remove)}")
    
    # =================================================================
    # CLEANUP: Remove failed model directories
    # =================================================================
    
    if not dirs_to_remove:
        print("\n✅ No directories to remove!")
        return
    
    print(f"\n🗑️ DIRECTORIES TO BE REMOVED:")
    print("-" * 30)
    for dir_path in dirs_to_remove:
        size_mb = get_folder_size(dir_path)
        print(f"   📁 {dir_path.name} ({size_mb:.1f} MB)")
    
    # Confirm with user
    print(f"\n⚠️ WARNING: This will permanently delete {len(dirs_to_remove)} directories!")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Cleanup cancelled by user")
        return
    
    print(f"\n🧹 REMOVING FAILED STATE MODEL DIRECTORIES...")
    print("-" * 45)
    
    removed_count = 0
    freed_space = 0
    
    for dir_path in dirs_to_remove:
        size_mb = get_folder_size(dir_path)
        print(f"🗑️ Removing {dir_path.name}...")
        
        if safe_remove_directory(dir_path):
            print(f"   ✅ Removed successfully ({size_mb:.1f} MB freed)")
            removed_count += 1
            freed_space += size_mb
        else:
            print(f"   ❌ Failed to remove")
    
    # =================================================================
    # CLEANUP: Remove old fix scripts (optional)
    # =================================================================
    
    print(f"\n🧹 CLEANING UP OLD FIX SCRIPTS...")
    print("-" * 35)
    
    # Root level fix scripts that are no longer needed
    fix_scripts = [
        "complete_fix_script.py",
        "final_training_fix.py", 
        "fix_cell_eval_genes.py",
        "fix_embedding_dimensions.py",
        "fix_gpu_issue.py",
        "fix_pytorch_loading.py",
        "fix_transformer_issue.py",
        "force_subset_training.py",
        "quick_fix_script.py",
        "quick_memory_fix.py",
        "quick_patch_script.py",
        "remove_early_stopping.py",
        "verify_setup_script.py"
    ]
    
    scripts_to_remove = []
    for script in fix_scripts:
        if os.path.exists(script):
            size_kb = os.path.getsize(script) / 1024
            scripts_to_remove.append((script, size_kb))
    
    if scripts_to_remove:
        print(f"📄 Old fix scripts found:")
        for script, size_kb in scripts_to_remove:
            print(f"   📄 {script} ({size_kb:.1f} KB)")
        
        response = input(f"\nRemove {len(scripts_to_remove)} old fix scripts? (y/N): ").strip().lower()
        
        if response == 'y':
            for script, size_kb in scripts_to_remove:
                try:
                    os.remove(script)
                    print(f"   ✅ Removed {script}")
                except Exception as e:
                    print(f"   ❌ Failed to remove {script}: {e}")
    else:
        print("   ✅ No old fix scripts found")
    
    # =================================================================
    # CLEANUP: Remove backup files (optional)
    # =================================================================
    
    print(f"\n🧹 CLEANING UP BACKUP FILES...")
    print("-" * 30)
    
    # Find backup files
    backup_patterns = [
        "*.backup",
        "*.backup.py", 
        "*_backup.py",
        "*_backup.yaml",
        "*.original_backup"
    ]
    
    backup_files = []
    for pattern in backup_patterns:
        backup_files.extend(glob.glob(pattern, recursive=True))
        # Also search in subdirectories
        backup_files.extend(glob.glob(f"**/{pattern}", recursive=True))
    
    # Remove duplicates and filter
    backup_files = list(set(backup_files))
    backup_files = [f for f in backup_files if os.path.exists(f)]
    
    if backup_files:
        print(f"📄 Backup files found:")
        total_backup_size = 0
        for backup_file in backup_files[:10]:  # Show first 10
            size_kb = os.path.getsize(backup_file) / 1024
            total_backup_size += size_kb
            print(f"   📄 {backup_file} ({size_kb:.1f} KB)")
        
        if len(backup_files) > 10:
            print(f"   ... and {len(backup_files) - 10} more files")
        
        print(f"   Total backup size: {total_backup_size:.1f} KB")
        
        response = input(f"\nRemove {len(backup_files)} backup files? (y/N): ").strip().lower()
        
        if response == 'y':
            removed_backups = 0
            for backup_file in backup_files:
                try:
                    os.remove(backup_file)
                    removed_backups += 1
                except Exception as e:
                    print(f"   ❌ Failed to remove {backup_file}: {e}")
            print(f"   ✅ Removed {removed_backups}/{len(backup_files)} backup files")
    else:
        print("   ✅ No backup files found")
    
    # =================================================================
    # SUMMARY
    # =================================================================
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print("=" * 30)
    print(f"✅ Directories removed: {removed_count}/{len(dirs_to_remove)}")
    print(f"💾 Space freed: {freed_space:.1f} MB ({freed_space/1024:.2f} GB)")
    
    print(f"\n📁 PRESERVED IMPORTANT DIRECTORIES:")
    for dir_name, description in keep_dirs.items():
        dir_path = outputs_dir / dir_name
        if dir_path.exists():
            size_mb = get_folder_size(dir_path)
            print(f"   ✅ {dir_name} ({size_mb:.1f} MB)")
    
    print(f"\n🚀 Your project is now cleaner!")
    print(f"   ✅ Failed model runs removed")
    print(f"   ✅ Successful models preserved") 
    print(f"   ✅ Important data files kept")
    print(f"   ✅ Ready for new experiments")

def main():
    """Main entry point"""
    
    # Change to project directory if needed
    project_root = Path(__file__).parent
    if project_root.name != "Virtual_Cell3":
        # Try to find the project root
        current = Path.cwd()
        while current.parent != current:
            if current.name == "Virtual_Cell3" or (current / "outputs").exists():
                os.chdir(current)
                break
            current = current.parent
    
    try:
        cleanup_state_models()
    except KeyboardInterrupt:
        print(f"\n❌ Cleanup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())