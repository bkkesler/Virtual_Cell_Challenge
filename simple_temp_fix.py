# Simple Temp Directory Fix for cell-eval
import os
import tempfile
import subprocess
import sys

print("🔧 SIMPLE TEMP DIRECTORY FIX")
print("=" * 30)

# Set custom temp directory
custom_temp = r"D:\temp_celleval"
os.makedirs(custom_temp, exist_ok=True)

# Set ALL possible temp environment variables
os.environ['TMPDIR'] = custom_temp
os.environ['TMP'] = custom_temp  
os.environ['TEMP'] = custom_temp
os.environ['TEMPDIR'] = custom_temp

# Also set Python's tempfile module
tempfile.tempdir = custom_temp

print(f"📁 Temp directory: {tempfile.gettempdir()}")

# Check space
def get_free_space_gb(path):
    import shutil
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)

free_space = get_free_space_gb(custom_temp)
print(f"💾 Available space: {free_space:.1f} GB")

# Files
h5ad_file = "esm2_go_rf_v2_submission.h5ad"
gene_names_file = "gene_names.txt"

print(f"\n📊 FILES:")
for file in [h5ad_file, gene_names_file]:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1e6
        print(f"✅ {file} ({size:.1f} MB)")
    else:
        print(f"❌ {file} - NOT FOUND")
        sys.exit(1)

# Create a Python script that sets temp and runs cell-eval
script_content = f'''
import os
import tempfile
import subprocess

# Set temp directory
os.environ['TMPDIR'] = r"{custom_temp}"
os.environ['TMP'] = r"{custom_temp}"
os.environ['TEMP'] = r"{custom_temp}"
tempfile.tempdir = r"{custom_temp}"

print(f"Using temp: {{tempfile.gettempdir()}}")

# Run cell-eval
cmd = ["python", "-m", "cell_eval", "prep", "-i", "{h5ad_file}", "--genes", "{gene_names_file}"]
print(f"Running: {{' '.join(cmd)}}")

result = subprocess.run(cmd, capture_output=False, text=True)
print(f"Exit code: {{result.returncode}}")
'''

# Write and run the script
script_file = "temp_celleval_runner.py"
with open(script_file, 'w') as f:
    f.write(script_content)

print(f"\n🚀 RUNNING CELL-EVAL WITH TEMP FIX:")
print("-" * 35)

try:
    # Run the script
    result = subprocess.run([sys.executable, script_file], 
                          capture_output=False, 
                          text=True,
                          timeout=600)  # 10 minute timeout
    
    print(f"\nProcess completed with exit code: {result.returncode}")
    
    if result.returncode == 0:
        print("✅ SUCCESS!")
    else:
        print("❌ Process failed")

except subprocess.TimeoutExpired:
    print("❌ Process timed out after 10 minutes")
except Exception as e:
    print(f"❌ Error: {e}")

# Check results
print(f"\n📦 CHECKING OUTPUT FILES:")
print("-" * 25)

vcc_files = [f for f in os.listdir('.') if f.endswith('.prep.vcc')]
if vcc_files:
    for file in vcc_files:
        size = os.path.getsize(file) / 1e6
        mtime = os.path.getmtime(file)
        print(f"📄 {file}: {size:.1f} MB (modified: {mtime})")
    
    latest = max(vcc_files, key=os.path.getmtime)
    print(f"\n🎯 Latest file: {latest}")
    
    # Check if it was created recently (last 10 minutes)
    import time
    if time.time() - os.path.getmtime(latest) < 600:
        print("✅ File was created recently - likely the new one!")
    else:
        print("⚠️ File is old - may be from previous attempt")
        
else:
    print("❌ No .prep.vcc files found")

# Cleanup
print(f"\n🧹 CLEANUP:")
print("-" * 10)

try:
    os.remove(script_file)
    print(f"✅ Removed temporary script")
except:
    pass

# Clean temp directory
try:
    temp_contents = os.listdir(custom_temp)
    for item in temp_contents:
        item_path = os.path.join(custom_temp, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)
        except:
            pass
    
    if temp_contents:
        print(f"✅ Cleaned {len(temp_contents)} temp items")
    else:
        print(f"✅ No temp files to clean")
        
except Exception as e:
    print(f"⚠️ Cleanup error: {e}")

print(f"\n🎯 If you see 'SUCCESS!' above and a recently created .prep.vcc file,")
print(f"you should now have a working VCC submission file!")
