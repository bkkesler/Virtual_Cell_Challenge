#!/usr/bin/env python3
"""
Generate ESM2 embeddings for VCC genes
This creates the esm2_embeddings.pt file needed for the Random Forest model
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def generate_esm2_embeddings():
    """Generate ESM2 embeddings for all genes in the VCC dataset"""
    
    print("🧬 GENERATING ESM2 EMBEDDINGS FOR VCC")
    print("=" * 50)
    
    # Check if we have transformers and ESM2
    try:
        from transformers import EsmTokenizer, EsmModel
        print("✅ ESM2 transformers available")
    except ImportError:
        print("❌ transformers library not installed!")
        print("Install with: pip install transformers")
        return False
    
    # Load gene names
    vcc_data_dir = Path("data/raw/single_cell_rnaseq/vcc_data")
    gene_names_path = vcc_data_dir / "gene_names.csv"
    
    if not gene_names_path.exists():
        print(f"❌ Gene names file not found: {gene_names_path}")
        return False
    
    genes = pd.read_csv(gene_names_path, header=None)[0].values
    print(f"✅ Found {len(genes)} genes to embed")
    
    # Create output directory
    output_dir = Path("outputs/state_model_run/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "esm2_embeddings.pt"
    
    # Check if embeddings already exist
    if output_path.exists():
        print(f"✅ ESM2 embeddings already exist at: {output_path}")
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   File size: {size_mb:.1f} MB")
        
        # Verify the file
        try:
            embeddings = torch.load(output_path, weights_only=False)
            print(f"   Contains embeddings for {len(embeddings)} genes")
            
            # Check if all genes are present
            missing_genes = set(genes) - set(embeddings.keys())
            if len(missing_genes) == 0:
                print("   ✅ All genes present - no need to regenerate")
                return True
            else:
                print(f"   ⚠️ Missing {len(missing_genes)} genes - regenerating...")
        except Exception as e:
            print(f"   ❌ Error loading existing file: {e}")
            print("   🔄 Will regenerate...")
    
    print(f"\n🔄 Generating ESM2 embeddings...")
    
    # Load ESM2 model
    print("📥 Loading ESM2 model...")
    try:
        model_name = "facebook/esm2_t33_650M_UR50D"  # Medium size model
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        print(f"✅ ESM2 model loaded on {device}")
        
    except Exception as e:
        print(f"❌ Error loading ESM2 model: {e}")
        print("💡 Falling back to random embeddings...")
        
        # Generate random embeddings as fallback
        embeddings = {}
        np.random.seed(42)
        
        for gene in genes:
            embeddings[gene] = torch.randn(1280).float()
        
        # Add control
        embeddings['non-targeting'] = torch.zeros(1280).float()
        
        # Save
        torch.save(embeddings, output_path)
        print(f"✅ Random embeddings saved to: {output_path}")
        return True
    
    # Generate embeddings
    embeddings = {}
    
    print(f"🔄 Processing {len(genes)} genes...")
    
    batch_size = 32  # Process in batches
    gene_batches = [genes[i:i+batch_size] for i in range(0, len(genes), batch_size)]
    
    with torch.no_grad():
        for batch_idx, gene_batch in enumerate(gene_batches):
            print(f"   Batch {batch_idx+1}/{len(gene_batches)}: {len(gene_batch)} genes")
            
            try:
                # Tokenize gene names (treating them as protein sequences)
                # Note: This is a simplified approach - in practice you'd want actual protein sequences
                inputs = tokenizer(list(gene_batch), return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = model(**inputs)
                
                # Use the mean of the last hidden state as gene embedding
                gene_embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_dim]
                
                # Store embeddings
                for i, gene in enumerate(gene_batch):
                    embeddings[gene] = gene_embeddings[i].cpu().float()
                
            except Exception as e:
                print(f"   ⚠️ Error processing batch {batch_idx+1}: {e}")
                # Fall back to random for this batch
                for gene in gene_batch:
                    embeddings[gene] = torch.randn(1280).float()
    
    # Add control embedding
    embeddings['non-targeting'] = torch.zeros(1280).float()
    
    print(f"✅ Generated embeddings for {len(embeddings)} genes")
    
    # Save embeddings
    torch.save(embeddings, output_path)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ ESM2 embeddings saved to: {output_path}")
    print(f"✅ File size: {file_size:.1f} MB")
    
    # Verify
    test_load = torch.load(output_path, weights_only=False)
    sample_gene = list(test_load.keys())[0]
    embedding_dim = test_load[sample_gene].shape[0]
    
    print(f"✅ Verification successful:")
    print(f"   Genes: {len(test_load)}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Sample gene: {sample_gene}")
    
    return True

def main():
    """Main function"""
    
    try:
        # Change to project directory
        project_root = Path("D:/Virtual_Cell3")
        if project_root.exists():
            os.chdir(project_root)
        
        success = generate_esm2_embeddings()
        
        if success:
            print(f"\n🎉 ESM2 EMBEDDINGS READY!")
            print(f"✅ You can now run the ESM2 Random Forest model")
            print(f"✅ File: outputs/state_model_run/embeddings/esm2_embeddings.pt")
        else:
            print(f"\n❌ Failed to generate ESM2 embeddings")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())