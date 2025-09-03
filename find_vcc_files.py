#!/usr/bin/env python3
"""
Find all .vcc files in the project and show their details
"""

import os
from pathlib import Path
from datetime import datetime

def find_vcc_files():
    """Find all .vcc files and show their details"""
    
    print("🔍 SEARCHING FOR VCC FILES")
    print("=" * 40)
    
    # Search patterns
    vcc_patterns = ["*.vcc", "*.prep.vcc"]
    
    all_vcc_files = []
    
    # Search entire project directory
    project_root = Path(".")
    
    for pattern in vcc_patterns:
        files = list(project_root.rglob(pattern))
        all_vcc_files.extend(files)
    
    # Remove duplicates and sort by modification time
    unique_files = list(set(all_vcc_files))
    unique_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not unique_files:
        print("❌ No .vcc files found!")
        print("\n💡 Expected locations:")
        print("   - vcc_esm2_rf_submission/esm2_rf_submission.prep.vcc")
        print("   - outputs/vcc_submission/predictions_sparse.prep.vcc")
        return []
    
    print(f"✅ Found {len(unique_files)} VCC files:")
    print()
    
    for i, vcc_file in enumerate(unique_files):
        # Get file info
        stat_info = vcc_file.stat()
        size_mb = stat_info.st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(stat_info.st_mtime)
        
        print(f"{i+1}. 📄 {vcc_file}")
        print(f"   💾 Size: {size_mb:.1f} MB")
        print(f"   📅 Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Try to identify what model this is from
        if "esm2" in str(vcc_file).lower():
            print(f"   🤖 Model: ESM2 Random Forest")
        elif "sparse" in str(vcc_file).lower():
            print(f"   🤖 Model: STATE (sparse version)")
        elif "backup" in str(vcc_file).lower():
            print(f"   🤖 Model: STATE (backup)")
        else:
            print(f"   🤖 Model: Unknown")
        
        # Check if it's the most recent
        if i == 0:
            print(f"   ⭐ MOST RECENT")
        
        print()
    
    return unique_files

def check_specific_locations():
    """Check specific expected locations"""
    
    print("🎯 CHECKING EXPECTED LOCATIONS")
    print("-" * 35)
    
    expected_locations = [
        # ESM2 Random Forest
        "vcc_esm2_rf_submission/esm2_rf_submission.prep.vcc",
        "vcc_esm2_rf_submission/esm2_rf_submission.vcc",
        
        # STATE model locations
        "outputs/vcc_submission/predictions_sparse.prep.vcc",
        "outputs/vcc_submission/predictions_backup.prep.vcc",
        "outputs/vcc_submission/predictions.prep.vcc",
        
        # Alternative locations
        "./esm2_rf_submission.prep.vcc",
        "./predictions.prep.vcc"
    ]
    
    found_files = []
    
    for location in expected_locations:
        path = Path(location)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(path.stat().st_mtime)
            
            print(f"✅ {location}")
            print(f"   Size: {size_mb:.1f} MB, Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            found_files.append(path)
        else:
            print(f"❌ {location}")
    
    return found_files

def show_file_contents_info(vcc_file):
    """Show basic info about VCC file contents"""
    
    print(f"\n🔍 ANALYZING: {vcc_file.name}")
    print("-" * 30)
    
    try:
        # Try to load as HDF5/AnnData
        import h5py
        with h5py.File(vcc_file, 'r') as f:
            print("✅ File format: HDF5")
            
            if 'X' in f and 'obs' in f:
                print("✅ Contains AnnData structure")
                
                # Get dimensions
                if 'shape' in f['X'].attrs:
                    shape = f['X'].attrs['shape']
                    print(f"   Data shape: {shape}")
                
                # Check obs for perturbations
                if 'target_gene' in f['obs']:
                    target_genes = f['obs']['target_gene'][:]
                    unique_perts = len(set(target_genes))
                    print(f"   Perturbations: {unique_perts}")
                
    except Exception as e:
        print(f"⚠️ Could not analyze file structure: {e}")

def main():
    """Main function"""
    
    print("🔍 VCC FILE FINDER")
    print("=" * 30)
    
    # Find all VCC files
    all_files = find_vcc_files()
    
    # Check specific expected locations
    expected_files = check_specific_locations()
    
    # Show summary
    print(f"\n📋 SUMMARY")
    print("-" * 15)
    
    if all_files:
        most_recent = all_files[0]
        print(f"🎯 Most recent VCC file:")
        print(f"   {most_recent}")
        
        size_mb = most_recent.stat().st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(most_recent.stat().st_mtime)
        
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        