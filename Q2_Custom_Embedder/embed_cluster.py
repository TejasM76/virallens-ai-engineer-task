import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Tuple, List
import pickle


BASELINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus(corpus_path: str) -> pd.DataFrame:
    df = pd.read_csv(corpus_path)
    return df


def generate_embeddings_baseline(texts: List[str], model_name: str = BASELINE_MODEL) -> np.ndarray:
    print(f"Loading baseline model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings


def perform_kmeans_clustering(embeddings: np.ndarray, n_clusters: int = 5) -> Tuple[np.ndarray, dict]:
    print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    metrics = {}
    if len(set(labels)) > 1:
        metrics['silhouette'] = silhouette_score(embeddings, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
        metrics['inertia'] = kmeans.inertia_
    
    print(f"Silhouette Score: {metrics.get('silhouette', 'N/A'):.4f}")
    print(f"Calinski-Harabasz Score: {metrics.get('calinski_harabasz', 'N/A'):.2f}")
    print(f"Davies-Bouldin Score: {metrics.get('davies_bouldin', 'N/A'):.4f}")
    
    return labels, metrics


def perform_hdbscan_clustering(embeddings: np.ndarray, min_cluster_size: int = 5) -> Tuple[np.ndarray, dict]:
    print(f"\nPerforming HDBSCAN clustering (min_cluster_size={min_cluster_size})...")
    
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    metrics = {}
    if n_clusters > 1:
        mask = labels != -1
        if sum(mask) > n_clusters:
            metrics['silhouette'] = silhouette_score(embeddings[mask], labels[mask])
            metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings[mask], labels[mask])
            metrics['davies_bouldin'] = davies_bouldin_score(embeddings[mask], labels[mask])
    
    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = n_noise
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")
    
    return labels, metrics


def compute_similarity_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    print("\nComputing similarity metrics...")
    
    metrics = {}
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    if len(unique_labels) < 2:
        return metrics
    
    intra_similarities = []
    for label in unique_labels:
        mask = labels == label
        if sum(mask) > 1:
            cluster_embeddings = embeddings[mask]
            norms = np.linalg.norm(cluster_embeddings, axis=1)
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            intra_similarities.append(std_norm / mean_norm if mean_norm > 0 else 0)
    
    metrics['avg_intra_cluster_variance'] = np.mean(intra_similarities) if intra_similarities else 0
    
    return metrics


def save_embeddings(embeddings: np.ndarray, path: str):
    np.save(path, embeddings)
    print(f"Saved embeddings to {path}")


def load_embeddings(path: str) -> np.ndarray:
    embeddings = np.load(path)
    return embeddings


def main():
    corpus_path = "corpus.csv"
    output_dir = "."
    
    print("Loading corpus...")
    df = load_corpus(corpus_path)
    texts = df['text'].tolist()
    
    print(f"\nProcessing {len(texts)} text chunks...")
    
    embeddings = generate_embeddings_baseline(texts)
    save_embeddings(embeddings, f"{output_dir}/embeddings_baseline.npy")
    
    labels_kmeans, metrics_kmeans = perform_kmeans_clustering(embeddings, n_clusters=5)
    
    labels_hdbscan, metrics_hdbscan = perform_hdbscan_clustering(embeddings, min_cluster_size=10)
    
    sim_metrics = compute_similarity_metrics(embeddings, labels_kmeans)
    
    all_metrics = {
        'kmeans': metrics_kmeans,
        'hdbscan': metrics_hdbscan,
        'similarity': sim_metrics
    }
    
    with open(f"{output_dir}/clustering_metrics.pkl", 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print("\nClustering complete!")
    return df, embeddings, labels_kmeans, all_metrics


if __name__ == "__main__":
    main()