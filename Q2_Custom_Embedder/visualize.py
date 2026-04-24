import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE
from typing import List
import os


def load_data(corpus_path: str, embeddings_path: str) -> tuple:
    df = pd.read_csv(corpus_path)
    embeddings = np.load(embeddings_path)
    return df, embeddings


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, df: pd.DataFrame, output_dir: str = "."):
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=labels, cmap='viridis', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE Visualization of Document Embeddings (K-Means)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(f"{output_dir}/tsne_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot")


def plot_umap(embeddings: np.ndarray, labels: np.ndarray, df: pd.DataFrame, output_dir: str = "."):
    print("Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=labels, cmap='viridis', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Cluster')
    plt.title('UMAP Visualization of Document Embeddings (K-Means)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig(f"{output_dir}/umap_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved UMAP plot")


def plot_cluster_distribution(labels: np.ndarray, df: pd.DataFrame, output_dir: str = "."):
    plt.figure(figsize=(10, 6))
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.bar(unique_labels, counts, color='steelblue')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Documents')
    plt.title('Cluster Distribution')
    plt.xticks(unique_labels)
    
    plt.savefig(f"{output_dir}/cluster_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster distribution plot")


def plot_source_distribution(labels: np.ndarray, df: pd.DataFrame, output_dir: str = "."):
    plt.figure(figsize=(12, 6))
    
    source_cluster = pd.DataFrame({'source': df['source'], 'cluster': labels})
    crosstab = pd.crosstab(source_cluster['source'], source_cluster['cluster'])
    
    crosstab.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.xlabel('Document Source')
    plt.ylabel('Number of Pages')
    plt.title('Cluster Distribution by Source Document')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/source_cluster_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved source-cluster distribution plot")


def plot_similarity_heatmap(embeddings: np.ndarray, df: pd.DataFrame, output_dir: str = ".", sample_size: int = 50):
    print("Computing similarity matrix...")
    
    indices = np.random.choice(len(embeddings), min(sample_size, len(embeddings)), replace=False)
    sample_embeddings = embeddings[indices]
    
    norms = sample_embeddings / np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
    similarity = np.dot(norms, norms.T)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity[:30, :30], cmap='viridis', square=True)
    plt.title('Document Similarity Matrix (Sample)')
    plt.xlabel('Document Index')
    plt.ylabel('Document Index')
    plt.savefig(f"{output_dir}/similarity_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity heatmap")


def plot_cluster_metrics(metrics: dict, output_dir: str = "."):
    if 'kmeans' in metrics:
        km = metrics['kmeans']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        if 'silhouette' in km:
            axes[0].bar(['Silhouette'], [km['silhouette']], color='steelblue')
            axes[0].set_title('Silhouette Score')
            axes[0].set_ylim(0, 1)
        
        if 'calinski_harabasz' in km:
            axes[1].bar(['Calinski-Harabasz'], [km['calinski_harabasz']], color='coral')
            axes[1].set_title('Calinski-Harabasz Score')
        
        if 'davies_bouldin' in km:
            axes[2].bar(['Davies-Bouldin'], [km['davies_bouldin']], color='seagreen')
            axes[2].set_title('Davies-Bouldin Score (lower is better)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cluster_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved cluster metrics plot")


def plot_text_length_distribution(df: pd.DataFrame, output_dir: str = "."):
    df['text_length'] = df['text'].str.len()
    
    plt.figure(figsize=(10, 6))
    plt.hist(df['text_length'], bins=50, color='steelblue', edgecolor='black')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Text Length Distribution')
    plt.savefig(f"{output_dir}/text_length_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved text length distribution")


def generate_all_visualizations(df: pd.DataFrame, embeddings: np.ndarray, labels: np.ndarray, 
                             metrics: dict, output_dir: str = "."):
    print("\nGenerating visualizations...")
    
    plot_tsne(embeddings, labels, df, output_dir)
    plot_umap(embeddings, labels, df, output_dir)
    plot_cluster_distribution(labels, df, output_dir)
    plot_source_distribution(labels, df, output_dir)
    plot_similarity_heatmap(embeddings, df, output_dir)
    plot_cluster_metrics(metrics, output_dir)
    plot_text_length_distribution(df, output_dir)
    
    print("All visualizations complete!")


if __name__ == "__main__":
    df, embeddings = load_data("corpus.csv", "embeddings_baseline.npy")
    
    from embed_cluster import perform_kmeans_clustering, perform_hdbscan_clustering
    
    labels_kmeans, _ = perform_kmeans_clustering(embeddings, n_clusters=5)
    labels_hdbscan, _ = perform_hdbscan_clustering(embeddings, min_cluster_size=10)
    
    metrics = {'kmeans': _, 'hdbscan': labels_hdbscan}
    
    generate_all_visualizations(df, embeddings, labels_kmeans, metrics, ".")