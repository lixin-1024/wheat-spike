"""
Batch phenotype clustering utilities.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors as mcolors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times New Roman PS", "DejaVu Serif"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
}


class PhenotypeClusterAnalyzer:
    """Cluster phenotype vectors and expose structured UI-ready artifacts."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)

    def cluster(self, samples: list[dict], output_dir: str) -> dict:
        if not samples:
            raise ValueError("No samples available for clustering")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        prepared = self._prepare_samples(samples)
        feature_matrix = prepared["feature_matrix"]
        scaled = prepared["scaled_matrix"]
        reduced = prepared["reduced_matrix"]
        image_names = prepared["image_names"]
        feature_names = prepared["feature_names"]

        n_clusters = max(2, min(self.n_clusters, len(samples)))
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = clustering.fit_predict(reduced)

        embedding = self._build_embedding(reduced, len(samples))
        linkage_matrix = linkage(np.asarray(reduced, dtype=float), method="ward")
        cluster_centers = self._compute_cluster_centers(labels, feature_matrix, n_clusters)
        silhouette = self._safe_silhouette(reduced, labels)
        dendrogram_data = self._build_dendrogram_payload(linkage_matrix, image_names)
        cluster_summaries = self._build_cluster_summaries(samples, labels, cluster_centers, feature_names)

        self._save_labels_csv(output_path / "clustering_results.csv", image_names, labels, embedding)
        self._save_centers_csv(output_path / "cluster_centers.csv", feature_names, cluster_centers)
        self._save_embedding_plot(output_path / "cluster_embedding.png", image_names, labels, embedding)
        self._save_dendrogram(output_path / "cluster_dendrogram.png", image_names, linkage_matrix)

        return {
            "method": "agglomerative_tsne",
            "labels": labels,
            "embedding": embedding,
            "image_names": image_names,
            "feature_names": feature_names,
            "cluster_centers": cluster_centers,
            "silhouette_score": silhouette,
            "clusters": cluster_summaries,
            "dendrogram": dendrogram_data,
            "cluster_options": {
                "current": int(n_clusters),
                "min": 2,
                "max": int(max(2, min(8, len(samples)))),
            },
            "files": {
                "labels_csv": str(output_path / "clustering_results.csv"),
                "centers_csv": str(output_path / "cluster_centers.csv"),
                "embedding_plot": str(output_path / "cluster_embedding.png"),
                "dendrogram_plot": str(output_path / "cluster_dendrogram.png"),
            },
        }

    def _prepare_samples(self, samples: list[dict]) -> dict:
        image_names = [sample["image"] for sample in samples]
        feature_names = list(samples[0]["feature_names"])
        feature_matrix = np.vstack([np.asarray(sample["features"], dtype=float) for sample in samples])

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

        return {
            "image_names": image_names,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
            "scaled_matrix": scaled,
            "reduced_matrix": reduced,
        }

    def _build_embedding(self, reduced: np.ndarray, sample_count: int) -> np.ndarray:
        if sample_count < 5:
            if reduced.shape[1] == 1:
                return np.column_stack([reduced[:, 0], np.zeros(sample_count, dtype=float)])
            return reduced[:, :2]

        perplexity = max(2, min(10, sample_count - 1))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
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

    def _build_cluster_summaries(
        self,
        samples: list[dict],
        labels: np.ndarray,
        cluster_centers: np.ndarray,
        feature_names: list[str],
    ) -> list[dict]:
        summaries = []

        for cluster_id in sorted(np.unique(labels).tolist()):
            members = [sample for index, sample in enumerate(samples) if int(labels[index]) == int(cluster_id)]
            if not members:
                continue

            center = cluster_centers[int(cluster_id)]
            representative = min(
                members,
                key=lambda sample: float(np.linalg.norm(np.asarray(sample["features"], dtype=float) - center)),
            )
            metrics = self._aggregate_metrics(members, feature_names)

            summaries.append(
                {
                    "cluster_id": int(cluster_id),
                    "sample_count": len(members),
                    "sample_names": [member["image"] for member in members],
                    "representative_image": representative.get("image_url"),
                    "representative_name": representative["image"],
                    "thumbnail_urls": [member.get("image_url") for member in members if member.get("image_url")],
                    "aggregate_metrics": metrics["means"],
                    "metric_ranges": metrics["ranges"],
                }
            )

        return summaries

    def _aggregate_metrics(self, samples: list[dict], feature_names: list[str]) -> dict:
        matrix = np.vstack([np.asarray(sample["features"], dtype=float) for sample in samples])
        means = {}
        ranges = {}
        for index, name in enumerate(feature_names):
            values = matrix[:, index]
            means[name] = float(np.mean(values))
            ranges[name] = {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        aliases = {
            # Backward-compatible metric aliases consumed by the web frontend.
            "spike_length": "mean_spike_length_cm",
            "spike_length_cm": "mean_spike_length_cm",
            "mean_spikelet_length": "mean_spikelet_length_mm",
            "mean_spikelet_length_mm": "mean_spikelet_length_mm",
            "mean_spikelet_width": "mean_spikelet_width_mm",
            "mean_spikelet_width_mm": "mean_spikelet_width_mm",
            "mean_attachment_angle": "mean_attachment_angle",
            "symmetry_index": "mean_symmetry_index",
            "centroid_offset": "mean_centroid_offset",
        }
        for source, alias in aliases.items():
            if source in means:
                means[alias] = means[source]

        return {"means": means, "ranges": ranges}

    def _build_dendrogram_payload(self, linkage_matrix: np.ndarray, image_names: list[str]) -> dict:
        sample_count = len(image_names)
        leaves_order = dendrogram(linkage_matrix, no_plot=True, labels=image_names)["leaves"]
        leaf_x = {
            leaf_index: 40 + order * 44
            for order, leaf_index in enumerate(leaves_order)
        }

        node_members = {index: [index] for index in range(sample_count)}
        node_heights = {index: 0.0 for index in range(sample_count)}
        nodes = []
        links = []
        root_id = None

        for merge_index, row in enumerate(linkage_matrix):
            left = int(row[0])
            right = int(row[1])
            height = float(row[2])
            node_id = sample_count + merge_index
            left_members = node_members[left]
            right_members = node_members[right]
            members = left_members + right_members
            x_value = float(np.mean([leaf_x[idx] for idx in members]))
            node_members[node_id] = members
            node_heights[node_id] = height
            root_id = node_id

            nodes.append(
                {
                    "id": node_id,
                    "x": x_value,
                    "y": height,
                    "height": height,
                    "sample_indices": members,
                    "sample_names": [image_names[idx] for idx in members],
                    "left": left,
                    "right": right,
                }
            )
            links.extend(
                [
                    {
                        "id": f"{left}-{node_id}",
                        "child": left,
                        "parent": node_id,
                        "x1": float(np.mean([leaf_x[idx] for idx in left_members])),
                        "y1": node_heights[left],
                        "x2": x_value,
                        "y2": height,
                    },
                    {
                        "id": f"{right}-{node_id}",
                        "child": right,
                        "parent": node_id,
                        "x1": float(np.mean([leaf_x[idx] for idx in right_members])),
                        "y1": node_heights[right],
                        "x2": x_value,
                        "y2": height,
                    },
                ]
            )

        leaf_nodes = [
            {
                "id": index,
                "x": leaf_x[index],
                "y": 0.0,
                "height": 0.0,
                "sample_indices": [index],
                "sample_names": [name],
            }
            for index, name in enumerate(image_names)
        ]

        return {
            "root_id": root_id,
            "leaves": [
                {
                    "id": index,
                    "name": image_names[index],
                    "x": leaf_x[index],
                }
                for index in leaves_order
            ],
            "nodes": leaf_nodes + nodes,
            "links": links,
            "max_height": float(max((node["height"] for node in nodes), default=1.0)),
        }

    def _save_labels_csv(self, csv_path: Path, image_names, labels, embedding):
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "cluster", "embed_x", "embed_y"])
            for image_name, label, point in zip(image_names, labels, embedding):
                writer.writerow([image_name, int(label), f"{point[0]:.6f}", f"{point[1]:.6f}"])

    def _save_centers_csv(self, csv_path: Path, feature_names, centers):
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cluster", *feature_names])
            for cluster_id, center in enumerate(centers):
                writer.writerow([cluster_id, *[f"{value:.6f}" for value in center]])

    def _save_embedding_plot(self, image_path: Path, image_names, labels, embedding):
        palette = [
            "#1f77b4",  # blue
            "#d62728",  # red
            "#2ca02c",  # green
            "#ff7f0e",  # orange
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#17becf",  # cyan
        ]
        with plt.rc_context(PAPER_RC):
            fig, ax = plt.subplots(figsize=(7.2, 5.6), facecolor="white")
            ax.set_facecolor("white")

            unique_labels = sorted({int(x) for x in labels.tolist()})
            for cluster_id in unique_labels:
                mask = labels == cluster_id
                color = palette[cluster_id % len(palette)]
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=58,
                    alpha=0.92,
                    c=color,
                    edgecolors="black",
                    linewidths=0.5,
                    label=f"Cluster {cluster_id + 1}",
                    zorder=3,
                )

            for name, point in zip(image_names, embedding):
                ax.annotate(
                    name,
                    xy=(float(point[0]), float(point[1])),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7.5,
                    color="black",
                    alpha=0.88,
                )

            ax.set_title("Phenotype Cluster Embedding", pad=10)
            ax.set_xlabel("Embedding Dimension 1")
            ax.set_ylabel("Embedding Dimension 2")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, color="#8f8f8f")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(image_path, facecolor="white", bbox_inches="tight")
            plt.close(fig)

    def _save_dendrogram(self, image_path: Path, image_names, linkage_matrix):
        with plt.rc_context(PAPER_RC):
            fig, ax = plt.subplots(figsize=(8.2, 5.8), facecolor="white")
            sample_count = len(image_names)
            heights = np.asarray(linkage_matrix[:, 2], dtype=float)
            h_min = float(np.min(heights)) if len(heights) else 0.0
            h_max = float(np.max(heights)) if len(heights) else 1.0
            if abs(h_max - h_min) < 1e-12:
                h_max = h_min + 1.0

            norm = mcolors.Normalize(vmin=h_min, vmax=h_max)
            cmap = cm.get_cmap("turbo")

            # node_id >= sample_count 表示内部合并节点
            node_height = {
                sample_count + idx: float(row[2])
                for idx, row in enumerate(linkage_matrix)
            }

            def branch_color_func(node_id: int) -> str:
                height = node_height.get(int(node_id), h_min)
                r, g, b, _ = cmap(norm(height))
                return mcolors.to_hex((r, g, b), keep_alpha=False)

            dendrogram(
                linkage_matrix,
                labels=image_names,
                leaf_rotation=40,
                leaf_font_size=8,
                color_threshold=0,
                above_threshold_color="#2f2f2f",
                link_color_func=branch_color_func,
                ax=ax,
            )
            ax.set_title("Hierarchical Clustering Dendrogram", pad=10)
            ax.set_ylabel("Ward Linkage Distance")
            ax.set_xlabel("Sample")
            ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.3, color="#8f8f8f")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Merge Height", fontsize=9)

            fig.tight_layout()
            fig.savefig(image_path, facecolor="white", bbox_inches="tight")
            plt.close(fig)
