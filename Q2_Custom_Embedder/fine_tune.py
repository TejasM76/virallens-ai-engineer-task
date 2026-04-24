"""
Fine-tune Embedder for Domain-Specific Legal Texts
This script fine-tunes a sentence-transformer model on the legal documents
to create a domain-specific embedder.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def create_training_pairs(texts, labels):
    """Create training pairs for contrastive learning"""
    examples = []
    for i, text in enumerate(texts):
        examples.append(InputExample(texts=[text], label=labels[i]))
    return examples


def fine_tune_embedder(
    corpus_path: str = "corpus.csv",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "./fine_tuned_model",
    epochs: int = 3,
    batch_size: int = 16
):
    """Fine-tune the embedding model"""
    
    print("="*50)
    print("Fine-tuning Domain-Specific Embedder")
    print("="*50)
    
    # Load corpus
    df = pd.read_csv(corpus_path)
    texts = df['text'].tolist()
    
    print(f"Loaded {len(texts)} text chunks")
    
    # Create pseudo-labels based on document source
    from sklearn.cluster import KMeans
    
    print("Creating pseudo-labels from TF-IDF clustering...")
    
    # Get embeddings first
    base_model = SentenceTransformer(model_name)
    base_embeddings = base_model.encode(texts, show_progress_bar=True)
    
    # Cluster to create pseudo-labels
    n_clusters = min(5, len(set(df['source'].unique()) if 'source' in df.columns else 5))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(base_embeddings)
    
    print(f"Created {len(set(labels))} pseudo-label clusters")
    
    # Create training examples
    train_examples = create_training_pairs(texts, labels)
    
    # Split data
    train_data, val_data = train_test_split(train_examples, test_size=0.1, random_state=42)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Load base model
    print(f"\nLoading base model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Configure training
    train_loss = losses.ContrastiveLoss(model=model)
    
    # Warmup steps
    warmup_steps = int(len(train_dataloader) * 0.1)
    
    print(f"\nFine-tuning for {epochs} epochs...")
    print(f"Training samples: {len(train_data)}")
    
    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True
    )
    
    print(f"\nModel saved to: {output_dir}")
    print(f"To use: SentenceTransformer('{output_dir}')")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune domain-specific embedder")
    parser.add_argument("--corpus", default="corpus.csv", help="Corpus CSV file")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Base model")
    parser.add_argument("--output", default="./fine_tuned_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    fine_tune_embedder(
        corpus_path=args.corpus,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )