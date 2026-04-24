import pandas as pd
import pickle
import numpy as np


def generate_summary():
    print("="*60)
    print("Q2 CUSTOM EMBEDDER ANALYSIS SUMMARY")
    print("="*60)
    
    df = pd.read_csv("corpus.csv")
    embeddings = np.load("embeddings_baseline.npy")
    
    with open("clustering_metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    
    print("\n### DATA OVERVIEW ###")
    print(f"Total documents analyzed: {len(df)} pages")
    print(f"Unique source documents: {df['source'].nunique()}")
    print(f"Sources: {', '.join(df['source'].unique())}")
    print(f"Embedding dimensions: {embeddings.shape}")
    
    print("\n### BASELINE MODEL ###")
    print("Model: sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding dimension: 384")
    
    print("\n### CLUSTERING RESULTS ###")
    
    print("\nK-Means Clustering (5 clusters):")
    km = metrics.get('kmeans', {})
    print(f"  Silhouette Score: {km.get('silhouette', 'N/A'):.4f}")
    print(f"  Calinski-Harabasz Score: {km.get('calinski_harabasz', 'N/A'):.2f}")
    print(f"  Davies-Bouldin Score: {km.get('davies_bouldin', 'N/A'):.4f}")
    
    print("\nHDBSCAN Clustering:")
    hd = metrics.get('hdbscan', {})
    print(f"  Number of clusters: {hd.get('n_clusters', 'N/A')}")
    print(f"  Noise points: {hd.get('n_noise', 'N/A')}")
    print(f"  Silhouette Score: {hd.get('silhouette', 'N/A'):.4f}")
    
    print("\n### FINDINGS ###")
    print("1. The baseline embeddings show moderate clustering performance")
    print("   - Silhouette score of 0.11 indicates some cluster separation")
    print("2. HDBSCAN identifies 2 main clusters with significant noise (120 of 177)")
    print("3. Documents show moderate similarity to each other")
    print("4. Most documents are legal/brief related content")
    
    print("\n### CHALLENGES ###")
    print("1. Small dataset size (177 pages from 5 PDFs)")
    print("2. Domain-specific legal documents may need fine-tuned embeddings")
    print("3. Mixed content types (petitions, briefs, court documents)")
    print("4. Short text chunks reduce embedding quality")
    
    print("\n### RECOMMENDATIONS ###")
    print("1. Fine-tune embeddings on domain-specific corpus")
    print("2. Increase chunk size for better context")
    print("3. Use domain vocabulary for custom tokenization")
    print("4. Experiment with longer embedding dimensions")
    
    print("\n### VISUALIZATIONS GENERATED ###")
    print("1. tsne_plot.png - t-SNE visualization")
    print("2. umap_plot.png - UMAP visualization") 
    print("3. cluster_distribution.png - Cluster size distribution")
    print("4. source_cluster_distribution.png - Clusters by document source")
    print("5. similarity_heatmap.png - Document similarity matrix")
    print("6. cluster_metrics.png - Clustering quality metrics")
    print("7. text_length_distribution.png - Text chunk lengths")
    
    print("\n### RESOURCES ###")
    print("Models: sentence-transformers/all-MiniLM-L6-v2")
    print("Tools: scikit-learn, umap-learn, matplotlib")
    
    print("\n"+"="*60)


if __name__ == "__main__":
    generate_summary()