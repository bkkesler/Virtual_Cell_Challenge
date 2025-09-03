"""
Find and verify ESM2 embeddings location
"""

import os
from pathlib import Path
import torch


def find_esm2_embeddings():
    """Find all ESM2 embedding files in the project"""

    print("🔍 Searching for ESM2 embeddings...")

    # Search patterns
    search_patterns = [
        "**/esm2_embeddings.pt",
        "**/embeddings.pt",
        "**/*embedding*.pt",
        "**/*esm*.pt"
    ]

    found_files = []

    # Search from project root
    project_root = Path.cwd()

    for pattern in search_patterns:
        matches = list(project_root.rglob(pattern))
        for match in matches:
            if match.is_file():
                found_files.append(match)

    # Remove duplicates
    found_files = list(set(found_files))

    print(f"\n📁 Found {len(found_files)} potential embedding files:")

    valid_embeddings = []

    for file_path in found_files:
        try:
            # Try to load and inspect
            embeddings = torch.load(file_path, weights_only=False)

            if isinstance(embeddings, dict):
                # Check if it looks like gene embeddings
                sample_keys = list(embeddings.keys())[:5]
                sample_key = sample_keys[0] if sample_keys else None

                if sample_key and isinstance(embeddings[sample_key], (list, tuple)) or hasattr(embeddings[sample_key],
                                                                                               'shape'):
                    embedding_dim = len(embeddings[sample_key]) if isinstance(embeddings[sample_key],
                                                                              (list, tuple)) else \
                    embeddings[sample_key].shape[0]

                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB

                    print(f"\n✅ {file_path}")
                    print(f"   📊 {len(embeddings)} embeddings")
                    print(f"   📏 Dimension: {embedding_dim}")
                    print(f"   💾 Size: {file_size:.1f} MB")
                    print(f"   🧬 Sample genes: {sample_keys}")

                    # Check if it has the control
                    if 'non-targeting' in embeddings:
                        print(f"   ✅ Has control embedding")

                    valid_embeddings.append({
                        'path': file_path,
                        'count': len(embeddings),
                        'dimension': embedding_dim,
                        'size_mb': file_size,
                        'has_control': 'non-targeting' in embeddings
                    })
                else:
                    print(f"\n⚠️  {file_path} - Invalid format")
            else:
                print(f"\n⚠️  {file_path} - Not a dictionary")

        except Exception as e:
            print(f"\n❌ {file_path} - Error loading: {e}")

    return valid_embeddings


def update_random_forest_paths(embeddings_info):
    """Update the random forest script with correct embedding paths"""

    if not embeddings_info:
        print("❌ No valid embeddings found to update paths")
        return False

    # Find the best embedding file (largest, with control)
    best_embedding = max(embeddings_info,
                         key=lambda x: (x['has_control'], x['count'], x['dimension']))

    print(f"\n🎯 Best embedding file:")
    print(f"   Path: {best_embedding['path']}")
    print(f"   Count: {best_embedding['count']}")
    print(f"   Dimension: {best_embedding['dimension']}")

    # Update random forest script
    rf_script_paths = [
        Path("models/random_forest/esm2_rf_model.py"),
        Path("scripts/random_forest/esm2_rf_model.py"),
        Path("src/models/random_forest/esm2_rf_model.py")
    ]

    rf_script = None
    for path in rf_script_paths:
        if path.exists():
            rf_script = path
            break

    if not rf_script:
        print("❌ Random forest script not found")
        return False

    print(f"\n🔧 Updating {rf_script}...")

    # Read the script
    with open(rf_script, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the embedding paths section and update it
    old_paths_pattern = r'embedding_paths = \[([^\]]+)\]'

    # Create new paths list with the correct path
    correct_path = str(best_embedding['path']).replace('\\', '/')
    new_paths = f'embedding_paths = [\n        r"{correct_path}"\n    ]'

    # Replace in content
    import re
    if re.search(old_paths_pattern, content):
        updated_content = re.sub(old_paths_pattern, new_paths, content)
    else:
        # If pattern not found, add at the beginning of main function
        main_pattern = r'def main\(\):'
        if re.search(main_pattern, content):
            updated_content = content.replace(
                'def main():',
                f'def main():\n    # Updated embedding path\n    {new_paths}'
            )
        else:
            print("⚠️  Could not find where to update paths")
            return False

    # Write back
    with open(rf_script, 'w', encoding='utf-8') as f:
        f.write(updated_content)

    print(f"✅ Updated random forest script with correct path")
    print(f"📁 Embedding path: {correct_path}")

    return True


def create_symlink_or_copy(best_embedding_path):
    """Create a copy of embeddings in expected location"""

    expected_path = Path("outputs/state_model_run/embeddings/esm2_embeddings.pt")
    expected_path.parent.mkdir(parents=True, exist_ok=True)

    if not expected_path.exists():
        import shutil
        shutil.copy2(best_embedding_path, expected_path)
        print(f"✅ Copied embeddings to expected location: {expected_path}")
        return True
    else:
        print(f"✅ Embeddings already exist at expected location")
        return True


def main():
    """Main function"""

    print("🔍 ESM2 Embeddings Finder")
    print("=" * 50)

    # Find all embedding files
    embeddings_info = find_esm2_embeddings()

    if not embeddings_info:
        print("\n❌ No valid ESM2 embeddings found!")
        print("\n💡 To create embeddings, run:")
        print("   python scripts/training/train_state_model.py --create-embeddings")
        return 1

    # Show summary
    print(f"\n📊 SUMMARY:")
    print(f"   Found {len(embeddings_info)} valid embedding files")

    best_embedding = max(embeddings_info,
                         key=lambda x: (x['has_control'], x['count'], x['dimension']))

    print(f"   Best file: {best_embedding['path']}")
    print(f"   Embeddings: {best_embedding['count']}")
    print(f"   Dimension: {best_embedding['dimension']}")

    # Update random forest script
    if update_random_forest_paths(embeddings_info):
        print(f"\n✅ Random forest script updated!")

    # Copy to expected location if needed
    create_symlink_or_copy(best_embedding['path'])

    print(f"\n🚀 Now you can run:")
    print(f"   python models/random_forest/esm2_rf_model.py")

    return 0


if __name__ == "__main__":
    exit(main())