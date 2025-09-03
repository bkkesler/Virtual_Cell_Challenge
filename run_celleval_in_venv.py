# Run cell-eval in Virtual Environment with Temp Fix
import os
import tempfile
import sys

print("🔧 SETTING UP TEMP DIRECTORY FOR CELL-EVAL")
print("=" * 45)

# Set custom temp directory
custom_temp = r"D:\temp_celleval"
os.makedirs(custom_temp, exist_ok=True)

# Override temp directory settings
os.environ['TMPDIR'] = custom_temp
os.environ['TMP'] = custom_temp  
os.environ['TEMP'] = custom_temp
tempfile.tempdir = custom_temp

print(f"📁 Temp directory set to: {tempfile.gettempdir()}")

# Check available space
def get_free_space_gb(path):
    import shutil
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)

free_space = get_free_space_gb(custom_temp)
print(f"💾 Available space: {free_space:.1f} GB")

# Now import and run cell_eval
print(f"\n🚀 IMPORTING AND RUNNING CELL-EVAL")
print("-" * 35)

try:
    # Import cell_eval CLI
    from cell_eval._cli._prep import prep
    import argparse
    
    print(f"✅ cell_eval imported successfully")
    print(f"🔍 Using temp directory: {tempfile.gettempdir()}")
    
    # Set up arguments as if called from command line
    class Args:
        input = "esm2_go_rf_v2_submission.h5ad"
        genes = "gene_names.txt"
        output = None  # Will auto-generate
        pert_col = None
        celltype_col = None
        ntc_name = None
        output_pert_col = None
        output_celltype_col = None
        encoding = None
        allow_discrete = False
        expected_gene_dim = None
        max_cell_dim = None
    
    args = Args()
    
    # Check input files
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.genes):
        print(f"❌ Gene file not found: {args.genes}")
        sys.exit(1)
    
    input_size = os.path.getsize(args.input) / 1e6
    print(f"✅ Input file: {args.input} ({input_size:.1f} MB)")
    print(f"✅ Gene file: {args.genes}")
    
    # Run the prep function
    print(f"\n⏳ Running cell-eval prep (may take 5-10 minutes)...")
    print(f"📊 This will create a .prep.vcc file")
    
    # Call the prep function directly
    prep(args)
    
    print(f"\n✅ cell-eval prep completed!")
    
    # Check for output files
    output_files = [f for f in os.listdir('vcc_esm2_go_rf_v2_submission') if f.endswith('.prep.vcc')]
    
    if output_files:
        latest_file = max(output_files, key=os.path.getmtime)
        size = os.path.getsize(latest_file) / 1e6
        print(f"📦 Created: {latest_file} ({size:.1f} MB)")
        
        # Compare with working submission
        working_file = "vcc_cross_dataset_submission/cross_dataset_full_submission_optimized.prep.vcc"
        if os.path.exists(working_file):
            working_size = os.path.getsize(working_file) / 1e6
            ratio = size / working_size
            print(f"📊 Size vs working submission: {ratio:.2f}x ({working_size:.1f} MB)")
            
            if 0.7 <= ratio <= 2.0:
                print(f"✅ Size ratio looks reasonable - file likely good!")
            else:
                print(f"⚠️ Unusual size ratio - double-check file")
        
        print(f"\n🚀 READY FOR VCC SUBMISSION!")
        print(f"📤 Upload: {latest_file}")
        
    else:
        print(f"❌ No .prep.vcc files created")

except ImportError as e:
    print(f"❌ Could not import cell_eval: {e}")
    print(f"💡 Make sure you're running this from your virtual environment")
    print(f"   Navigate to: {os.getcwd()}")
    print(f"   Activate venv: .venv\\Scripts\\activate")
    print(f"   Then run: python run_celleval_in_venv.py")

except Exception as e:
    print(f"❌ Error running cell-eval: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup temp files
    print(f"\n🧹 CLEANING UP TEMP FILES")
    print("-" * 25)
    
    try:
        temp_files = os.listdir(custom_temp)
        if temp_files:
            for item in temp_files:
                item_path = os.path.join(custom_temp, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        print(f"🗑️ Removed: {item}")
                    elif os.path.isdir(item_path):
                        import shutil
                        shutil.rmtree(item_path)
                        print(f"🗑️ Removed directory: {item}")
                except Exception as e:
                    print(f"⚠️ Could not remove {item}: {e}")
            print(f"✅ Cleanup completed")
        else:
            print(f"✅ No temp files to clean")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

print(f"\n🎯 SUMMARY:")
print("-" * 10)
print("If successful, you now have a properly created .prep.vcc file")
print("without disk space errors, ready for VCC submission.")
