# Direct Python Temp Directory Fix for cell-eval
import os
import tempfile
import subprocess
import sys

print("🔧 DIRECT TEMP DIRECTORY FIX FOR CELL-EVAL")
print("=" * 45)

# Set custom temp directory BEFORE importing anything else
custom_temp = r"D:\temp_celleval"
os.makedirs(custom_temp, exist_ok=True)

# Override ALL possible temp directory settings
os.environ['TMPDIR'] = custom_temp
os.environ['TMP'] = custom_temp  
os.environ['TEMP'] = custom_temp
tempfile.tempdir = custom_temp

print(f"📁 Custom temp directory: {custom_temp}")
print(f"🔍 Python will use: {tempfile.gettempdir()}")

# Check available space
def get_free_space_gb(path):
    import shutil
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)

free_space = get_free_space_gb(custom_temp)
print(f"💾 Available space: {free_space:.1f} GB")

if free_space < 10:
    print(f"⚠️ WARNING: Less than 10GB free space")
    print(f"   Consider freeing up more space on D: drive")

# Files to process
h5ad_file = "esm2_go_rf_v2_submission.h5ad"
gene_names_file = "gene_names.txt"

print(f"\n📊 PROCESSING FILES:")
print("-" * 20)

# Check files exist
if not os.path.exists(h5ad_file):
    print(f"❌ File not found: {h5ad_file}")
    sys.exit(1)

if not os.path.exists(gene_names_file):
    print(f"❌ File not found: {gene_names_file}")
    sys.exit(1)

input_size = os.path.getsize(h5ad_file) / 1e6
print(f"✅ Input file: {h5ad_file} ({input_size:.1f} MB)")
print(f"✅ Gene names: {gene_names_file}")

# Estimated temp space needed (3-4x input size)
estimated_temp_needed = input_size * 3.5 / 1000  # Convert to GB
print(f"📊 Estimated temp space needed: {estimated_temp_needed:.1f} GB")

if free_space < estimated_temp_needed:
    print(f"❌ Insufficient space! Need {estimated_temp_needed:.1f} GB, have {free_space:.1f} GB")
    sys.exit(1)

print(f"✅ Sufficient space available")

# Run cell-eval with fixed temp directory
print(f"\n🚀 RUNNING CELL-EVAL WITH FIXED TEMP:")
print("-" * 40)

cmd = ["python", "-m", "cell_eval", "prep", "-i", h5ad_file, "--genes", gene_names_file]
print(f"Command: {' '.join(cmd)}")
print(f"Temp dir: {tempfile.gettempdir()}")

try:
    # Start the process
    print(f"\n⏳ Starting cell-eval (this may take 5-10 minutes)...")
    
    # Use Popen to show real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line.strip())
    
    # Wait for completion
    return_code = process.wait()
    
    print(f"\n📊 PROCESS COMPLETED:")
    print(f"Return code: {return_code}")
    
    if return_code == 0:
        print(f"✅ SUCCESS!")
        
        # Check for output files
        output_files = [f for f in os.listdir('.') if f.endswith('.prep.vcc')]
        
        if output_files:
            for file in output_files:
                size = os.path.getsize(file) / 1e6
                print(f"📦 Created: {file} ({size:.1f} MB)")
                
                # Quick integrity check
                if size > 100:  # Should be at least 100MB
                    print(f"✅ File size looks reasonable")
                else:
                    print(f"⚠️ File size seems small - may be corrupted")
        else:
            print(f"❌ No .prep.vcc files found")
    else:
        print(f"❌ Process failed with return code: {return_code}")

except KeyboardInterrupt:
    print(f"\n❌ Process interrupted by user")
    if 'process' in locals():
        process.terminate()
except Exception as e:
    print(f"❌ Error running cell-eval: {e}")

# Cleanup temp files
print(f"\n🧹 CLEANING UP TEMP FILES:")
print("-" * 30)

try:
    temp_files = os.listdir(custom_temp)
    cleaned_size = 0
    
    for item in temp_files:
        item_path = os.path.join(custom_temp, item)
        try:
            if os.path.isfile(item_path):
                file_size = os.path.getsize(item_path)
                os.remove(item_path)
                cleaned_size += file_size
                print(f"🗑️ Removed file: {item}")
            elif os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)
                print(f"🗑️ Removed directory: {item}")
        except Exception as e:
            print(f"⚠️ Could not remove {item}: {e}")
    
    if cleaned_size > 0:
        print(f"✅ Cleaned up {cleaned_size / 1e6:.1f} MB of temp files")
    else:
        print(f"✅ No temp files to clean up")
        
except Exception as e:
    print(f"⚠️ Cleanup error: {e}")

# Final status
print(f"\n🎯 FINAL STATUS:")
print("-" * 15)

vcc_files = [f for f in os.listdir('.') if f.endswith('.prep.vcc')]
if vcc_files:
    latest_vcc = max(vcc_files, key=os.path.getmtime)
    vcc_size = os.path.getsize(latest_vcc) / 1e6
    
    print(f"✅ VCC file ready: {latest_vcc}")
    print(f"📊 Size: {vcc_size:.1f} MB")
    print(f"🚀 Ready for VCC submission!")
    
    # Compare with previous working submission
    prev_working = "../vcc_cross_dataset_submission/cross_dataset_full_submission_optimized.prep.vcc"
    if os.path.exists(prev_working):
        prev_size = os.path.getsize(prev_working) / 1e6
        ratio = vcc_size / prev_size
        print(f"📈 Size vs previous working: {ratio:.2f}x ({prev_size:.1f} MB)")
        
        if 0.5 <= ratio <= 2.0:
            print(f"✅ Size ratio looks reasonable")
        else:
            print(f"⚠️ Size ratio unusual - double-check file")
else:
    print(f"❌ No VCC files created")
    print(f"Check the output above for errors")

print(f"\n💡 KEY DIFFERENCE FROM MANUAL APPROACH:")
print("This script sets tempfile.tempdir BEFORE running cell-eval,")
print("ensuring Python uses D: drive for ALL temporary operations.")
