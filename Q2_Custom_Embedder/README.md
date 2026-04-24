# Q2: Custom Embedder for Domain Documents - Virallens Assignment

## Overview
This project implements a domain-specific embedding analysis on legal documents using sentence-transformers with clustering, fine-tuning, and visualization.

## Approach

### 1. Preprocessing
- Extract text from 5 PDF documents using PyPDF
- 177 pages total from legal briefs and court documents
- Text cleaned and normalized

### 2. Baseline Embedding
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- 384-dimensional embeddings
- Pre-trained model for general text understanding

### 3. Fine-tuning (Domain-Specific)
- Fine-tune the baseline model on legal domain texts
- Use contrastive loss for similarity learning
- Creates domain-specific embeddings

### 4. Clustering
- **K-Means**: 5 clusters
- **HDBSCAN**: Dense clustering

### 5. Evaluation Metrics
- Silhouette Score (cluster cohesion)
- Calinski-Harabasz Score
- Davies-Bouldin Score

## Results

### Baseline K-Means Clustering
| Metric | Score |
|--------|-------|
| Silhouette | 0.1144 |
| Calinski-Harabasz | 14.12 |
| Davies-Bouldin | 2.5218 |

### Baseline HDBSCAN
| Metric | Value |
|--------|-------|
| Clusters | 2 |
| Noise Points | 120 |

## Running

### Step 1: Extract text
```bash
python preprocessing.py
```

### Step 2: Generate baseline embeddings & cluster
```bash
python embed_cluster.py
```

### Step 3: Fine-tune for domain-specific embeddings
```bash
python fine_tune.py --epochs 5
```

### Step 4: Visualize
```bash
python visualize.py
```

### Step 5: Generate summary
```bash
python summary.py
```

## Visualizations Generated (7 total)
1. tsne_plot.png - t-SNE projection
2. umap_plot.png - UMAP projection
3. cluster_distribution.png - Cluster sizes
4. source_cluster_distribution.png - Source by cluster
5. similarity_heatmap.png - Document similarity
6. cluster_metrics.png - Quality metrics plot
7. text_length_distribution.png - Text lengths

## Fine-tuning Details
The fine-tuning script uses contrastive learning to adapt the baseline model to legal domain texts:
- Base model: sentence-transformers/all-MiniLM-L6-v2
- Training: Contrastive loss with pseudo-labels from clustering
- Output: Domain-specific embedding model in ./fine_tuned_model

Expected Improvement after fine-tuning:
- Silhouette Score: ~20-40% improvement
- Better cluster separation for legal documents

## Findings
1. Moderate baseline clustering performance (Silhouette: 0.11)
2. High noise with HDBSCAN (120/177 documents)
3. Domain-specific fine-tuning should significantly improve results

## Challenges
- Small dataset (177 pages)
- Mixed legal content types
- Short text chunks reduce embedding quality
- Fine-tuning recommended for production use

## Resources
- Model: sentence-transformers/all-MiniLM-L6-v2
- Fine-tuning: sentence-transformers library
- Clustering: scikit-learn, HDBSCAN
- Visualization: matplotlib, seaborn, umap-learn