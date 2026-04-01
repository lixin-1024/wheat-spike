"""
批量表型聚类分析模块。

相较于简单 KMeans + PCA，这里改为：
1. 标准化特征
2. PCA 预降维以抑制噪声
3. 层次聚类（Agglomerative）
4. 使用 t-SNE 生成二维嵌入用于可视化
5. 额外输出样本相似度热图与树状图
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class PhenotypeClusterAnalyzer:
    """执行批量聚类并输出更完整的可视化结果。"""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)

    def cluster(self, samples: list[dict], output_dir: str) -> dict:
        if not samples:
            raise ValueError("没有可用于聚类的样本")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_names = [sample['image'] for sample in samples]
        feature_names = list(samples[0]['feature_names'])
        feature_matrix = np.vstack([np.asarray(sample['features'], dtype=float) for sample in samples])

        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_matrix)

        max_components = min(scaled.shape[1], max(1, scaled.shape[0] - 1))
        pca_dims = min(max(2, min(10, scaled.shape[1])), max_components)
        if np.allclose(np.var(scaled, axis=0), 0.0):
            reduced = scaled[:, :max(1, min(2, scaled.shape[1]))]
            if reduced.shape[1] == 1:
                reduced = np.column_stack([reduced[:, 0], np.zeros(len(samples), dtype=float)])
        else:
            reduced = PCA(n_components=max(1, pca_dims), random_state=self.random_state).fit_transform(scaled)
            if reduced.shape[1] == 1:
                reduced = np.column_stack([reduced[:, 0], np.zeros(len(samples), dtype=float)])

        n_clusters = max(2, min(self.n_clusters, len(samples)))
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = clustering.fit_predict(reduced)

        embedding = self._build_embedding(reduced, len(samples))
        distance_matrix = squareform(pdist(scaled, metric='euclidean'))
        linkage_matrix = linkage(np.asarray(reduced, dtype=float), method='ward')
        cluster_centers = self._compute_cluster_centers(labels, feature_matrix, n_clusters)
        silhouette = self._safe_silhouette(reduced, labels)

        self._save_labels_csv(output_path / "clustering_results.csv", image_names, labels, embedding)
        self._save_centers_csv(output_path / "cluster_centers.csv", feature_names, cluster_centers)
        self._save_embedding_plot(output_path / "cluster_embedding.png", image_names, labels, embedding)
        self._save_heatmap(output_path / "sample_similarity_heatmap.png", image_names, distance_matrix)
        self._save_dendrogram(output_path / "cluster_dendrogram.png", image_names, linkage_matrix)

        return {
            'method': 'agglomerative_tsne',
            'labels': labels,
            'embedding': embedding,
            'image_names': image_names,
            'feature_names': feature_names,
            'cluster_centers': cluster_centers,
            'silhouette_score': silhouette,
            'files': {
                'labels_csv': str(output_path / "clustering_results.csv"),
                'centers_csv': str(output_path / "cluster_centers.csv"),
                'embedding_plot': str(output_path / "cluster_embedding.png"),
                'heatmap_plot': str(output_path / "sample_similarity_heatmap.png"),
                'dendrogram_plot': str(output_path / "cluster_dendrogram.png"),
            },
        }

    def _build_embedding(self, reduced: np.ndarray, sample_count: int) -> np.ndarray:
        if sample_count < 3:
            if reduced.shape[1] == 1:
                return np.column_stack([reduced[:, 0], np.zeros(sample_count, dtype=float)])
            return reduced[:, :2]

        if sample_count < 5:
            if reduced.shape[1] == 1:
                return np.column_stack([reduced[:, 0], np.zeros(sample_count, dtype=float)])
            return reduced[:, :2]

        perplexity = max(2, min(10, sample_count - 1))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            init='pca',
            random_state=self.random_state,
        )
        return tsne.fit_transform(reduced)

    def _compute_cluster_centers(self, labels: np.ndarray, feature_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        centers = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            centers.append(feature_matrix[mask].mean(axis=0) if np.any(mask) else np.zeros(feature_matrix.shape[1]))
        return np.vstack(centers)

    def _safe_silhouette(self, reduced: np.ndarray, labels: np.ndarray):
        if len(np.unique(labels)) < 2 or len(labels) < 3:
            return None
        return float(silhouette_score(reduced, labels))

    def _save_labels_csv(self, csv_path: Path, image_names, labels, embedding):
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['image', 'cluster', 'embed_x', 'embed_y'])
            for image_name, label, point in zip(image_names, labels, embedding):
                writer.writerow([image_name, int(label), f"{point[0]:.6f}", f"{point[1]:.6f}"])

    def _save_centers_csv(self, csv_path: Path, feature_names, centers):
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['cluster', *feature_names])
            for cluster_id, center in enumerate(centers):
                writer.writerow([cluster_id, *[f"{value:.6f}" for value in center]])

    def _save_embedding_plot(self, image_path: Path, image_names, labels, embedding):
        plt.figure(figsize=(8.6, 6.6), facecolor='#08111f')
        ax = plt.gca()
        ax.set_facecolor('#08111f')
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap='viridis',
            s=84,
            alpha=0.9,
            edgecolors='#dff7ff',
            linewidths=0.8,
        )
        for name, point in zip(image_names, embedding):
            ax.text(point[0] + 0.8, point[1] + 0.8, name, fontsize=8, color='#dff7ff', alpha=0.88)
        ax.set_title("Phenotype Cluster Map", color='white', fontsize=14, pad=12)
        ax.set_xlabel("Embedding X", color='#dff7ff')
        ax.set_ylabel("Embedding Y", color='#dff7ff')
        ax.tick_params(colors='#9dc9ff')
        ax.grid(color='#19324f', alpha=0.5)
        cbar = plt.colorbar(scatter)
        cbar.ax.yaxis.set_tick_params(color='#dff7ff')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#dff7ff')
        plt.tight_layout()
        plt.savefig(image_path, dpi=220, facecolor='#08111f')
        plt.close()

    def _save_heatmap(self, image_path: Path, image_names, distance_matrix):
        plt.figure(figsize=(7.8, 6.4), facecolor='white')
        plt.imshow(distance_matrix, cmap='magma')
        plt.title("Sample Distance Heatmap")
        plt.xticks(range(len(image_names)), image_names, rotation=45, ha='right', fontsize=8)
        plt.yticks(range(len(image_names)), image_names, fontsize=8)
        plt.colorbar(label='Euclidean Distance')
        plt.tight_layout()
        plt.savefig(image_path, dpi=220)
        plt.close()

    def _save_dendrogram(self, image_path: Path, image_names, linkage_matrix):
        plt.figure(figsize=(8.6, 6.2), facecolor='white')
        dendrogram(linkage_matrix, labels=image_names, leaf_rotation=45, leaf_font_size=8)
        plt.title("Phenotype Hierarchical Clustering")
        plt.ylabel("Ward Distance")
        plt.tight_layout()
        plt.savefig(image_path, dpi=220)
        plt.close()
