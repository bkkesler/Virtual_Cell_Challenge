# Check and Install cell-eval in Virtual Environment
import subprocess
import sys
import os

print("🔍 CHECKING CELL-EVAL INSTALLATION")
print("=" * 40)

# Check if we're in a virtual environment
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("✅ Running in virtual environment")
    print(f"📁 Python path: {sys.executable}")
    print(f"📁 Environment: {sys.prefix}")
else:
    print("⚠️ Not in virtual environment - this might be the issue!")

# Check current working directory
print(f"📁 Current directory: {os.getcwd()}")

# Method 1: Check if cell-eval is importable
print(f"\n🧪 TEST 1: Import cell_eval module")
print("-" * 30)
try:
    import cell_eval
    print(f"✅ cell_eval module found")
    print(f"📍 Location: {cell_eval.__file__}")
    print(f"📋 Version: {getattr(cell_eval, '__version__', 'Unknown')}")
except ImportError as e:
    print(f"❌ cell_eval module not found: {e}")
    print("🔧 Need to install cell-eval")

# Method 2: Check if cell-eval command is available
print(f"\n🧪 TEST 2: cell-eval command")
print("-" * 25)
try:
    result = subprocess.run(['cell-eval', '--version'], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print(f"✅ cell-eval command works")
        print(f"📋 Output: {result.stdout.strip()}")
    else:
        print(f"❌ cell-eval command failed: {result.stderr}")
except FileNotFoundError:
    print(f"❌ cell-eval command not found")
except Exception as e:
    print(f"❌ Error testing cell-eval: {e}")

# Method 3: Check if python -m cell_eval works
print(f"\n🧪 TEST 3: python -m cell_eval")
print("-" * 30)
try:
    result = subprocess.run(['python', '-m', 'cell_eval', '--version'], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print(f"✅ python -m cell_eval works")
        print(f"📋 Output: {result.stdout.strip()}")
    else:
        print(f"❌ python -m cell_eval failed")
        print(f"📋 Error: {result.stderr.strip()}")
except Exception as e:
    print(f"❌ Error testing python -m cell_eval: {e}")

# Check what's installed in the environment
print(f"\n🧪 TEST 4: Check installed packages")
print("-" * 35)
try:
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        cell_eval_lines = [line for line in lines if 'cell' in line.lower() or 'eval' in line.lower()]
        
        if cell_eval_lines:
            print(f"📦 Found cell/eval related packages:")
            for line in cell_eval_lines:
                print(f"   {line}")
        else:
            print(f"❌ No cell-eval related packages found")
            
        # Also check for common related packages
        print(f"\n📦 Other relevant packages:")
        relevant_packages = ['anndata', 'scanpy', 'pandas', 'numpy', 'scipy']
        for line in lines:
            for pkg in relevant_packages:
                if line.lower().startswith(pkg.lower()):
                    print(f"   {line}")
                    break
    else:
        print(f"❌ Could not list packages: {result.stderr}")
except Exception as e:
    print(f"❌ Error listing packages: {e}")

# Try to install cell-eval if not found
print(f"\n🔧 INSTALLATION CHECK")
print("-" * 20)

# Check if any tests passed
cell_eval_works = False
try:
    import cell_eval
    cell_eval_works = True
except:
    pass

if not cell_eval_works:
    print(f"❌ cell-eval not properly installed")
    print(f"💡 SOLUTIONS:")
    print(f"   1. Install cell-eval:")
    print(f"      pip install cell-eval")
    print(f"   2. Or try:")
    print(f"      pip install --upgrade cell-eval")
    print(f"   3. Or install from source:")
    print(f"      pip install git+https://github.com/Genentech/cell-eval.git")
    
    # Try to install automatically
    response = input(f"\n❓ Try to install cell-eval automatically? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print(f"\n🔄 Installing cell-eval...")
        try:
            result = subprocess.run(['pip', 'install', 'cell-eval'], 
                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"✅ cell-eval installation successful!")
                print(f"📋 Output: {result.stdout[-500:]}")  # Last 500 chars
                
                # Test again
                try:
                    import cell_eval
                    print(f"✅ cell-eval now imports successfully!")
                except ImportError:
                    print(f"❌ cell-eval still not importable after installation")
                    
            else:
                print(f"❌ Installation failed:")
                print(f"📋 Error: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Installation error: {e}")
    else:
        print(f"⏭️ Skipping automatic installation")

else:
    print(f"✅ cell-eval is properly installed!")
    
    # If cell-eval works, create the working command
    print(f"\n🚀 CREATING WORKING CELL-EVAL COMMAND")
    print("-" * 40)
    
    # Set up temp directory
    custom_temp = r"D:\temp_celleval"
    os.makedirs(custom_temp, exist_ok=True)
    
    # Create a working script
    working_script = '''
import os
import tempfile
import subprocess
import sys

# Set temp directory
custom_temp = r"D:\\temp_celleval"
os.environ['TMPDIR'] = custom_temp
os.environ['TMP'] = custom_temp
os.environ['TEMP'] = custom_temp
tempfile.tempdir = custom_temp

print(f"🔧 Using temp directory: {tempfile.gettempdir()}")

# Check files
files = ["esm2_go_rf_v2_submission.h5ad", "gene_names.txt"]
for file in files:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1e6
        print(f"✅ {file} ({size:.1f} MB)")
    else:
        print(f"❌ {file} not found")
        sys.exit(1)

# Run cell-eval
cmd = ["python", "-m", "cell_eval", "prep", "-i", "esm2_go_rf_v2_submission.h5ad", "--genes", "gene_names.txt"]
print(f"🚀 Running: {' '.join(cmd)}")

try:
    result = subprocess.run(cmd, timeout=600)
    print(f"📊 Exit code: {result.returncode}")
    
    if result.returncode == 0:
        print(f"✅ SUCCESS! Check for .prep.vcc file")
    else:
        print(f"❌ Failed with exit code: {result.returncode}")
        
except subprocess.TimeoutExpired:
    print(f"❌ Timed out after 10 minutes")
except Exception as e:
    print(f"❌ Error: {e}")
'''
    
    with open("run_working_celleval.py", 'w') as f:
        f.write(working_script)
    
    print(f"✅ Created: run_working_celleval.py")
    print(f"📋 Usage: python run_working_celleval.py")

print(f"\n🎯 SUMMARY:")
print("-" * 10)
print(f"If cell-eval is installed, run: python run_working_celleval.py")
print(f"If not installed, run: pip install cell-eval")
print(f"Then try the script again.")
