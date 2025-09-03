#!/usr/bin/env python3
"""
Prepare ESM2 Random Forest submission for VCC using cell-eval prep
"""

import os
import sys
import subprocess
from pathlib import Path

def prep_esm2_submission():
    """Run cell-eval prep on the ESM2 Random Forest submission"""
    
    print("📦 PREPARING ESM2 RF SUBMISSION FOR VCC")
    print("=" * 50)
    
    # Paths for ESM2 Random Forest submission
    submission_dir = Path("vcc_esm2_rf_submission")
    h5ad_path = submission_dir / "esm2_rf_submission.h5ad"
    gene_names_path = submission_dir / "gene_names.txt"
    
    # Check if files exist
    if not h5ad_path.exists():
        print(f"❌ Submission file not found: {h5ad_path}")
        print("Please run the ESM2 Random Forest model first!")
        return False
    
    if not gene_names_path.exists():
        print(f"❌ Gene names file not found: {gene_names_path}")
        return False
    
    # Show file info
    file_size_mb = h5ad_path.stat().st_size / (1024 * 1024)
    print(f"✅ Found submission files:")
    print(f"   📄 H5AD file: {h5ad_path}")
    print(f"   📄 Gene names: {gene_names_path}")
    print(f"   💾 Size: {file_size_mb:.1f} MB")
    
    # Check if cell-eval is available
    print(f"\n🔍 Checking cell-eval availability...")
    
    try:
        # Test if cell-eval is installed
        result = subprocess.run([sys.executable, "-m", "cell_eval", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ cell-eval is available")
        else:
            print("❌ cell-eval not properly installed")
            return False
    except Exception as e:
        print(f"❌ Error checking cell-eval: {e}")
        return False
    
    # Run cell-eval prep
    print(f"\n🚀 Running cell-eval prep...")
    print(f"   Input: {h5ad_path}")
    print(f"   Genes: {gene_names_path}")
    
    cmd = [
        sys.executable, "-m", "cell_eval", "prep",
        "-i", str(h5ad_path),
        "--genes", str(gene_names_path)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"⏳ This may take several minutes for a {file_size_mb:.0f} MB file...")
    
    try:
        # Run with longer timeout for large files
        result = subprocess.run(cmd, capture_output=True, text=True, 
                               check=True, timeout=1200)  # 20 minute timeout
        
        print("✅ cell-eval prep completed successfully!")
        
        if result.stdout:
            print("\n📊 STDOUT:")
            print(result.stdout)
        
        # Find the output .prep.vcc file
        possible_outputs = [
            h5ad_path.with_suffix('.prep.vcc'),
            h5ad_path.parent / f"{h5ad_path.stem}.prep.vcc",
            submission_dir / "esm2_rf_submission.prep.vcc"
        ]
        
        output_file = None
        for output_path in possible_outputs:
            if output_path.exists():
                output_file = output_path
                break
        
        if output_file:
            final_size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"\n🎉 SUCCESS!")
            print(f"✅ VCC submission file created: {output_file}")
            print(f"✅ Final size: {final_size_mb:.1f} MB")
            
            print(f"\n📊 ESM2 RANDOM FOREST SUBMISSION READY:")
            print(f"   🎪 Model: Random Forest with ESM2 embeddings")
            print(f"   🧬 Embeddings: 1280D protein embeddings")
            print(f"   📊 Training: Pseudo-bulk profiles (151 perturbations)")
            print(f"   🎯 Expected performance: High DE score, good MAE")
            print(f"   📤 File to upload: {output_file.name}")
            
            print(f"\n🚀 NEXT STEPS:")
            print(f"   1. Upload {output_file.name} to VCC platform")
            print(f"   2. Compare results with STATE model (Overall: 0.007)")
            print(f"   3. Update model tracker with official scores")
            
            return str(output_file)
        else:
            print("❌ No .prep.vcc output file found")
            print("Check if cell-eval completed successfully")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ cell-eval prep failed: {e}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        
        # If prep fails, the .h5ad might still be valid for direct upload
        print(f"\n💡 ALTERNATIVE:")
        print(f"   📄 Original .h5ad file is still valid: {h5ad_path}")
        print(f"   🔧 Try uploading {h5ad_path.name} directly to VCC")
        print(f"   📞 Or contact VCC support if issues persist")
        
        return str(h5ad_path)
        
    except subprocess.TimeoutExpired:
        print(f"❌ cell-eval prep timed out (20 minutes)")
        print(f"   Large file size ({file_size_mb:.1f} MB) may need more time")
        print(f"   Try running manually: python -m cell_eval prep -i {h5ad_path} --genes {gene_names_path}")
        return None

def main():
    """Main function"""
    
    try:
        # Change to project directory
        if Path("D:/Virtual_Cell3").exists():
            os.chdir("D:/Virtual_Cell3")
        
        result = prep_esm2_submission()
        
        if result:
            if result.endswith('.prep.vcc'):
                print(f"\n🎉 ESM2 RANDOM FOREST READY FOR VCC!")
                print(f"📤 Upload: {Path(result).name}")
            else:
                print(f"\n⚠️ Partial success - try direct upload")
            return 0
        else:
            print(f"\n❌ Failed to prepare submission")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())