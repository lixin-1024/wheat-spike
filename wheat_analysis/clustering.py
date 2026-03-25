"""
批量麦穗特征聚类分析。
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class SpikeClusterAnalyzer:
    """基于表型特征向量执行聚类并可视化。"""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)

    def cluster(self, samples: list[dict], output_dir: str) -> dict:
        """
        Args:
            samples: [{'image': str, 'feature_names': list[str], 'features': np.ndarray}, ...]
        """
        if not samples:
            raise ValueError("没有可用于聚类的样本")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        feature_names = list(samples[0]['feature_names'])
        image_names = [sample['image'] for sample in samples]
        feature_matrix = np.vstack([np.asarray(sample['features'], dtype=float) for sample in samples])

        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_matrix)

        n_clusters = min(self.n_clusters, len(samples))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(scaled)

        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(scaled)
        centers_2d = pca.transform(kmeans.cluster_centers_)

        self._save_labels_csv(output_path / "clustering_results.csv", image_names, labels, points_2d)
        self._save_centers_csv(output_path / "cluster_centers.csv", feature_names, scaler.inverse_transform(kmeans.cluster_centers_))
        self._save_pca_plot(output_path / "clustering_pca.png", image_names, labels, points_2d, centers_2d)

        return {
            'labels': labels,
            'pca_points': points_2d,
            'feature_names': feature_names,
            'cluster_centers': scaler.inverse_transform(kmeans.cluster_centers_),
            'explained_variance_ratio': pca.explained_variance_ratio_,
        }

    def _save_labels_csv(self, csv_path: Path, image_names, labels, points_2d):
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['image', 'cluster', 'pca_x', 'pca_y'])
            for image_name, label, point in zip(image_names, labels, points_2d):
                writer.writerow([image_name, int(label), f"{point[0]:.6f}", f"{point[1]:.6f}"])

    def _save_centers_csv(self, csv_path: Path, feature_names, centers):
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['cluster', *feature_names])
            for idx, center in enumerate(centers):
                writer.writerow([idx, *[f"{value:.6f}" for value in center]])

    def _save_pca_plot(self, image_path: Path, image_names, labels, points_2d, centers_2d):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1], c=labels, cmap='tab10', s=55, alpha=0.85)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', marker='x', s=120, linewidths=2)

        for image_name, point in zip(image_names, points_2d):
            plt.text(point[0] + 0.03, point[1] + 0.03, image_name, fontsize=8, alpha=0.8)

        plt.title("Wheat Spike Clustering (PCA)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True, alpha=0.25)
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig(image_path, dpi=200)
        plt.close()
