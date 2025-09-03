# Working cell-eval Script
import os
import tempfile
import subprocess
import sys

print("Working cell-eval with temp directory fix")
print("=" * 45)

# Set temp directory to D drive
custom_temp = r"D:\temp_celleval"
os.makedirs(custom_temp, exist_ok=True)

# Set environment variables
os.environ['TMPDIR'] = custom_temp
os.environ['TMP'] = custom_temp
os.environ['TEMP'] = custom_temp
tempfile.tempdir = custom_temp

print(f"Temp directory set to: {tempfile.gettempdir()}")

# Check files exist
files = ["esm2_go_rf_v2_submission.h5ad", "gene_names.txt"]
for file in files:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1e6
        print(f"Found {file} ({size:.1f} MB)")
    else:
        print(f"ERROR: {file} not found")
        sys.exit(1)

# Check available space
def get_free_space_gb(path):
    import shutil
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)

free_space = get_free_space_gb(custom_temp)
print(f"Available space: {free_space:.1f} GB")

if free_space < 10:
    print("WARNING: Less than 10GB free space")

print("\nRunning cell-eval prep...")

# Use the cell-eval command directly since it works
cmd = ["cell-eval", "prep", "-i", "esm2_go_rf_v2_submission.h5ad", "--genes", "gene_names.txt"]
print(f"Command: {' '.join(cmd)}")

try:
    # Run the command with real-time output
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
        print(line.rstrip())
    
    # Wait for completion
    return_code = process.wait()
    
    print(f"\nProcess completed with exit code: {return_code}")
    
    if return_code == 0:
        print("SUCCESS!")
        
        # Check for output files
        vcc_files = [f for f in os.listdir('.') if f.endswith('.prep.vcc')]
        if vcc_files:
            latest_file = max(vcc_files, key=os.path.getmtime)
            size = os.path.getsize(latest_file) / 1e6
            print(f"Created: {latest_file} ({size:.1f} MB)")
            
            # Check if created recently
            import time
            if time.time() - os.path.getmtime(latest_file) < 300:  # 5 minutes
                print("File was created recently - this is your new submission file!")
            else:
                print("File is older - may be from previous attempt")
                
        else:
            print("No .prep.vcc files found")
    else:
        print(f"FAILED with exit code: {return_code}")

except subprocess.TimeoutExpired:
    print("ERROR: Process timed out")
    process.terminate()
except Exception as e:
    print(f"ERROR: {e}")

# Cleanup temp files
print("\nCleaning up temp files...")
try:
    temp_files = os.listdir(custom_temp)
    for item in temp_files:
        item_path = os.path.join(custom_temp, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)
        except:
            pass
    
    if temp_files:
        print(f"Cleaned up {len(temp_files)} temp items")
    else:
        print("No temp files to clean")
        
except Exception as e:
    print(f"Cleanup error: {e}")

print("\nDone! If you see SUCCESS above, you should have a working .prep.vcc file")
