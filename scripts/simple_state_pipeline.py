"""
Simple STATE pipeline that works with current file structure
"""

import os
import sys
from pathlib import Path

def main():
    print("🧬 Simple STATE Pipeline")
    project_root = Path.cwd()
    print(f"Project root: {project_root}")
    
    # Check for training script
    possible_training_scripts = [
        project_root / "scripts" / "training" / "train_state_model.py",
        project_root / "scripts" / "train_state_model.py",
        project_root / "scripts" / "train_state_minimal.py"
    ]
    
    training_script = None
    for script in possible_training_scripts:
        if script.exists():
            training_script = script
            break
    
    if training_script:
        print(f"✅ Found training script: {training_script.relative_to(project_root)}")
        
        # Try to run it with minimal arguments
        cmd = [
            sys.executable, str(training_script),
            "--help"
        ]
        
        print("Running: " + " ".join(cmd))
        os.system(" ".join(cmd))
    else:
        print("❌ No training script found")
        print("Available scripts:")
        scripts_dir = project_root / "scripts"
        if scripts_dir.exists():
            for item in scripts_dir.rglob("*.py"):
                print(f"  - {item.relative_to(project_root)}")

if __name__ == "__main__":
    main()
