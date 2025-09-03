# Perturbation Gene Comparison Across Submissions
import os
import pandas as pd
import anndata as ad
import numpy as np
from pathlib import Path

print("🔍 PERTURBATION GENE COMPARISON ACROSS SUBMISSIONS")
print("=" * 55)

# Define all your submission directories and files
submissions = {
    "v2_current": ("./vcc_esm2_go_rf_v2_submission", "esm2_go_rf_v2_submission.h5ad"),
    "v1_original": ("./vcc_esm2_go_rf_submission", "esm2_go_rf_submission.h5ad"),
    "differential": ("./vcc_esm2_rf_differential_submission", "esm2_rf_differential_submission.h5ad"),
    "cross_dataset": ("./vcc_cross_dataset_submission", "cross_dataset_full_submission_optimized.h5ad"),
}

# Also check the original VCC validation file
vcc_validation_file = "./data/raw/single_cell_rnaseq/vcc_data/pert_counts_Validation.csv"

print("📊 LOADING PERTURBATION DATA FROM ALL SUBMISSIONS")
print("-" * 50)

perturbation_data = {}
file_info = {}

# Load perturbations from each submission
for name, (directory, filename) in submissions.items():
    filepath = os.path.join(directory, filename)
    
    if os.path.exists(filepath):
        try:
            print(f"📁 Loading {name}: {filename}")
            adata = ad.read_h5ad(filepath)
            
            if 'target_gene' in adata.obs.columns:
                pert_counts = adata.obs['target_gene'].value_counts()
                perturbation_data[name] = set(pert_counts.index)
                
                file_info[name] = {
                    'shape': adata.shape,
                    'perturbations': len(pert_counts),
                    'total_cells': adata.n_obs,
                    'pert_counts': pert_counts
                }
                
                print(f"   ✅ {len(pert_counts)} perturbations, {adata.n_obs:,} cells")
                
                # Show top perturbations
                print(f"   Top 5: {list(pert_counts.head().index)}")
                
                # Check for non-targeting
                if 'non-targeting' in pert_counts:
                    print(f"   ✅ Non-targeting: {pert_counts['non-targeting']:,} cells")
                else:
                    print(f"   ❌ Non-targeting not found!")
                    
            else:
                print(f"   ❌ No 'target_gene' column found")
                file_info[name] = {'error': 'No target_gene column'}
                
        except Exception as e:
            print(f"   ❌ Error loading {filepath}: {e}")
            file_info[name] = {'error': str(e)}
    else:
        print(f"   ❌ File not found: {filepath}")
        file_info[name] = {'error': 'File not found'}

# Load original VCC validation perturbations
print(f"\n📋 Loading original VCC validation data...")
if os.path.exists(vcc_validation_file):
    try:
        vcc_validation = pd.read_csv(vcc_validation_file)
        expected_perturbations = set(vcc_validation['target_gene'].tolist())
        
        # Add non-targeting if not present
        expected_perturbations.add('non-targeting')
        
        perturbation_data['vcc_expected'] = expected_perturbations
        file_info['vcc_expected'] = {
            'perturbations': len(expected_perturbations),
            'source': 'pert_counts_Validation.csv'
        }
        
        print(f"   ✅ Expected perturbations: {len(expected_perturbations)}")
        print(f"   Sample: {list(list(expected_perturbations)[:5])}")
        
    except Exception as e:
        print(f"   ❌ Error loading validation file: {e}")
else:
    print(f"   ❌ VCC validation file not found: {vcc_validation_file}")

print(f"\n🔍 DETAILED PERTURBATION COMPARISON")
print("-" * 40)

# Compare all submissions
if len(perturbation_data) > 1:
    # Find common perturbations across all submissions
    all_perturbations = set()
    for perts in perturbation_data.values():
        all_perturbations.update(perts)
    
    print(f"📊 PERTURBATION OVERLAP ANALYSIS:")
    print(f"   Total unique perturbations across all files: {len(all_perturbations)}")
    
    # Create comparison matrix
    comparison_data = []
    
    for name1, perts1 in perturbation_data.items():
        row = {'submission': name1, 'total_perts': len(perts1)}
        
        for name2, perts2 in perturbation_data.items():
            if name1 != name2:
                overlap = len(perts1 & perts2)
                total_union = len(perts1 | perts2)
                jaccard = overlap / total_union if total_union > 0 else 0
                
                row[f'overlap_{name2}'] = overlap
                row[f'jaccard_{name2}'] = f"{jaccard:.3f}"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(f"\n📈 OVERLAP MATRIX:")
    
    # Show simplified overlap matrix
    overlap_matrix = {}
    for name1, perts1 in perturbation_data.items():
        overlap_matrix[name1] = {}
        for name2, perts2 in perturbation_data.items():
            if name1 == name2:
                overlap_matrix[name1][name2] = len(perts1)
            else:
                overlap_matrix[name1][name2] = len(perts1 & perts2)
    
    # Print matrix
    names = list(perturbation_data.keys())
    print(f"{'':15} " + " ".join(f"{name:12}" for name in names))
    for name1 in names:
        values = " ".join(f"{overlap_matrix[name1][name2]:12}" for name2 in names)
        print(f"{name1:15} {values}")

# Identify missing/extra perturbations
print(f"\n🎯 MISSING/EXTRA PERTURBATIONS ANALYSIS")
print("-" * 45)

if 'vcc_expected' in perturbation_data:
    expected = perturbation_data['vcc_expected']
    
    for name, perts in perturbation_data.items():
        if name != 'vcc_expected':
            missing = expected - perts
            extra = perts - expected
            
            print(f"\n📋 {name.upper()}:")
            print(f"   Expected: {len(expected)}")
            print(f"   Actual: {len(perts)}")
            print(f"   Missing: {len(missing)}")
            print(f"   Extra: {len(extra)}")
            
            if missing:
                print(f"   ❌ Missing perturbations: {sorted(list(missing))}")
            
            if extra:
                print(f"   ⚠️ Extra perturbations: {sorted(list(extra))}")
            
            if not missing and not extra:
                print(f"   ✅ Perfect match with expected perturbations!")

# Check cell count distributions
print(f"\n📊 CELL COUNT DISTRIBUTION COMPARISON")
print("-" * 40)

for name, info in file_info.items():
    if 'pert_counts' in info:
        pert_counts = info['pert_counts']
        
        print(f"\n📈 {name.upper()}:")
        print(f"   Total cells: {info['total_cells']:,}")
        print(f"   Perturbations: {info['perturbations']}")
        print(f"   Cells per perturbation:")
        print(f"     Min: {pert_counts.min():,}")
        print(f"     Max: {pert_counts.max():,}")
        print(f"     Mean: {pert_counts.mean():.0f}")
        print(f"     Median: {pert_counts.median():.0f}")
        
        # Check for extremely imbalanced distributions
        max_min_ratio = pert_counts.max() / pert_counts.min()
        if max_min_ratio > 10:
            print(f"   ⚠️ High imbalance ratio: {max_min_ratio:.1f}x")
        else:
            print(f"   ✅ Reasonable balance ratio: {max_min_ratio:.1f}x")

# Generate specific recommendations
print(f"\n💡 RECOMMENDATIONS BASED ON ANALYSIS")
print("-" * 40)

recommendations = []

# Check if current submission matches expected
if 'vcc_expected' in perturbation_data and 'v2_current' in perturbation_data:
    expected = perturbation_data['vcc_expected']
    current = perturbation_data['v2_current']
    
    if current == expected:
        recommendations.append("✅ Current submission has correct perturbations")
    else:
        missing = expected - current
        extra = current - expected
        
        if missing:
            recommendations.append(f"❌ CRITICAL: Missing {len(missing)} expected perturbations")
            recommendations.append(f"   Missing: {sorted(list(missing))}")
            recommendations.append("   → This could cause silent submission failure")
        
        if extra:
            recommendations.append(f"⚠️ Extra {len(extra)} unexpected perturbations")
            recommendations.append(f"   Extra: {sorted(list(extra))}")

# Check if other submissions worked and had different perturbations
working_submissions = ['cross_dataset', 'differential']  # Assuming these worked
for working in working_submissions:
    if working in perturbation_data and 'v2_current' in perturbation_data:
        working_perts = perturbation_data[working]
        current_perts = perturbation_data['v2_current']
        
        if working_perts != current_perts:
            missing_from_current = working_perts - current_perts
            extra_in_current = current_perts - working_perts
            
            if missing_from_current or extra_in_current:
                recommendations.append(f"🔄 Perturbations differ from working {working} submission")
                if missing_from_current:
                    recommendations.append(f"   Missing vs {working}: {sorted(list(missing_from_current))}")
                if extra_in_current:
                    recommendations.append(f"   Extra vs {working}: {sorted(list(extra_in_current))}")

# Cell count recommendations
if 'v2_current' in file_info and 'pert_counts' in file_info['v2_current']:
    pert_counts = file_info['v2_current']['pert_counts']
    
    # Check for very low cell counts
    low_count_perts = pert_counts[pert_counts < 100]
    if len(low_count_perts) > 0:
        recommendations.append(f"⚠️ {len(low_count_perts)} perturbations have <100 cells")
        recommendations.append(f"   Low count perturbations: {dict(low_count_perts)}")

# Print recommendations
if recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
else:
    recommendations.append("✅ No obvious perturbation-related issues found")
    print("1. ✅ No obvious perturbation-related issues found")

print(f"\n🎯 IMMEDIATE ACTIONS")
print("-" * 20)

if 'vcc_expected' in perturbation_data and 'v2_current' in perturbation_data:
    expected = perturbation_data['vcc_expected']
    current = perturbation_data['v2_current']
    missing = expected - current
    
    if missing:
        print("🚨 CRITICAL ISSUE FOUND:")
        print(f"   Your submission is missing {len(missing)} required perturbations")
        print(f"   Missing: {sorted(list(missing))}")
        print(f"\n🔧 SOLUTION:")
        print("   1. Regenerate submission with all required perturbations")
        print("   2. Check your model training included all validation perturbations")
        print("   3. Verify pert_counts_Validation.csv was used correctly")
    else:
        print("✅ Perturbations match expected - issue is likely elsewhere")
        print("   Continue with browser/platform troubleshooting")

# Save detailed comparison report
report_path = "perturbation_comparison_report.txt"
with open(report_path, 'w') as f:
    f.write("PERTURBATION COMPARISON REPORT\n")
    f.write("=" * 35 + "\n\n")
    
    f.write("SUBMISSION SUMMARY:\n")
    for name, info in file_info.items():
        f.write(f"\n{name}:\n")
        if 'error' in info:
            f.write(f"  Error: {info['error']}\n")
        else:
            f.write(f"  Shape: {info.get('shape', 'N/A')}\n")
            f.write(f"  Perturbations: {info.get('perturbations', 'N/A')}\n")
            f.write(f"  Total cells: {info.get('total_cells', 'N/A')}\n")
    
    f.write(f"\nRECOMMENDATIONS:\n")
    for rec in recommendations:
        f.write(f"- {rec}\n")

print(f"\n📄 Detailed report saved: {report_path}")
print(f"🎯 This analysis will help identify if perturbation mismatches are causing your submission issues!")
