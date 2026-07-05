"""Candidate-level separability evaluation after preprocessing."""

from __future__ import annotations

from pathlib import Path

import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from common import (
    OUTPUT_ROOT,
    apply_grid,
    apply_robust_xlim,
    apply_robust_ylim,
    configure_matplotlib,
    ensure_dir,
    enhance_signal,
    extract_candidate_features,
    harmonize_record,
    load_image_records,
    robust_limits,
    robust_scale_from_images,
    save_figure,
)


SEPARABILITY_ROOT = OUTPUT_ROOT / "04_separability_evaluation"
FEATURE_COLUMNS = [
    "area",
    "eccentricity",
    "major_axis_length",
    "minor_axis_length",
    "axis_ratio",
    "solidity",
    "mean_intensity",
    "max_intensity",
    "enhanced_mean",
    "local_background_mean",
    "local_background_std",
    "sbr",
    "cnr",
]
UMAP_DOT_SIZE = 14
PANEL_TITLE_FONTSIZE = 17
GROUP_MARKERS = {"Bacteria": "o", "Particles": "X"}


def compute_reference_anchors() -> tuple[float, float]:
    """Compute Machine 1 reference anchors from baseline-corrected bacteria images.

    Args:
        None.

    Returns:
        tuple[float, float]: Low and high robust intensity anchors.
    """

    records = load_image_records()
    corrected = []
    for record in records:
        if record.group == "Bacteria":
            preview = harmonize_record(record, reference_low=0.0, reference_high=1.0)
            corrected.append(preview.baseline_corrected)
    return robust_scale_from_images(corrected, low_pct=5.0, high_pct=99.0)


def collect_candidate_features() -> pd.DataFrame:
    """Run preprocessing and collect candidate features for all images.

    Args:
        None.

    Returns:
        pd.DataFrame: Candidate-level features for every image.
    """

    reference_low, reference_high = compute_reference_anchors()
    rows: list[dict[str, float | str]] = []
    for record in load_image_records():
        harmonized_result = harmonize_record(record, reference_low=reference_low, reference_high=reference_high)
        _, enhanced = enhance_signal(harmonized_result.harmonized)
        features, _ = extract_candidate_features(
            enhanced,
            harmonized_result.harmonized,
            record.path.name,
            record.group,
            record.machine,
        )
        rows.extend(features)
    if not rows:
        raise RuntimeError("No candidate features were extracted; inspect enhancement thresholds.")
    return pd.DataFrame(rows)


def cohen_d(first: pd.Series, second: pd.Series) -> float:
    """Compute absolute Cohen's d effect size between two feature distributions.

    Args:
        first (pd.Series): First numeric distribution.
        second (pd.Series): Second numeric distribution.

    Returns:
        float: Absolute standardized mean difference.
    """

    first_values = first.dropna().astype(float).to_numpy()
    second_values = second.dropna().astype(float).to_numpy()
    if len(first_values) < 2 or len(second_values) < 2:
        return 0.0
    pooled_variance = ((len(first_values) - 1) * np.var(first_values, ddof=1)) + (
        (len(second_values) - 1) * np.var(second_values, ddof=1)
    )
    pooled_variance /= max(len(first_values) + len(second_values) - 2, 1)
    return float(abs(np.mean(first_values) - np.mean(second_values)) / max(np.sqrt(pooled_variance), 1e-9))


def quantile_overlap(first: pd.Series, second: pd.Series) -> float:
    """Estimate feature overlap using interquantile interval intersection.

    Args:
        first (pd.Series): First numeric distribution.
        second (pd.Series): Second numeric distribution.

    Returns:
        float: Overlap fraction between 5th-95th percentile ranges, where 0 is separated.
    """

    first_low, first_high = np.percentile(first.dropna().astype(float), [5, 95])
    second_low, second_high = np.percentile(second.dropna().astype(float), [5, 95])
    intersection = max(0.0, min(first_high, second_high) - max(first_low, second_low))
    union = max(first_high, second_high) - min(first_low, second_low)
    return float(intersection / max(union, 1e-9))


def build_separability_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Build numeric feature separability summary between bacteria and particles.

    Args:
        features (pd.DataFrame): Candidate feature table.

    Returns:
        pd.DataFrame: Feature effect sizes and overlap estimates.
    """

    bacteria = features[features["group"] == "Bacteria"]
    particles = features[features["group"] == "Particles"]
    rows = []
    for column in FEATURE_COLUMNS:
        rows.append(
            {
                "feature": column,
                "bacteria_median": float(bacteria[column].median()),
                "particles_median": float(particles[column].median()),
                "absolute_cohen_d": cohen_d(bacteria[column], particles[column]),
                "quantile_overlap_5_95": quantile_overlap(bacteria[column], particles[column]),
            }
        )
    return pd.DataFrame(rows).sort_values("absolute_cohen_d", ascending=False)


def finite_metric_value(value: float) -> float:
    """Convert metric values to plain finite floats for CSV output.

    Args:
        value (float): Metric value.

    Returns:
        float: Finite metric value, or NaN when the value is not finite.
    """

    return float(value) if np.isfinite(value) else float("nan")


def encoded_group_labels(groups: pd.Series) -> np.ndarray:
    """Encode bacteria and particle group labels as integers.

    Args:
        groups (pd.Series): Group labels with values such as Bacteria and Particles.

    Returns:
        np.ndarray: Integer labels where Bacteria and Particles map to stable numeric IDs.
    """

    return groups.map({"Bacteria": 0, "Particles": 1}).astype(int).to_numpy()


def valid_internal_cluster_labels(labels: np.ndarray) -> np.ndarray:
    """Return mask for samples that can be used in internal cluster metrics.

    Args:
        labels (np.ndarray): Cluster labels, where DBSCAN noise may be -1.

    Returns:
        np.ndarray: Boolean mask excluding noise labels.
    """

    return labels >= 0


def can_score_internal_metrics(coords: np.ndarray, labels: np.ndarray) -> bool:
    """Check whether internal clustering metrics are mathematically defined.

    Args:
        coords (np.ndarray): Two-dimensional embedding coordinates.
        labels (np.ndarray): Labels used for scoring.

    Returns:
        bool: True when at least two non-singleton clusters are available.
    """

    if coords.shape[0] < 3:
        return False
    unique_labels = np.unique(labels)
    return 1 < len(unique_labels) < coords.shape[0]


def internal_embedding_metrics(coords: np.ndarray, labels: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute standard internal separation metrics on an embedding.

    Args:
        coords (np.ndarray): Two-dimensional embedding coordinates.
        labels (np.ndarray): Labels to evaluate on the embedding.
        prefix (str): Prefix used for metric names.

    Returns:
        dict[str, float]: Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.
    """

    if not can_score_internal_metrics(coords, labels):
        return {
            f"{prefix}_silhouette": float("nan"),
            f"{prefix}_davies_bouldin": float("nan"),
            f"{prefix}_calinski_harabasz": float("nan"),
        }
    return {
        f"{prefix}_silhouette": finite_metric_value(silhouette_score(coords, labels)),
        f"{prefix}_davies_bouldin": finite_metric_value(davies_bouldin_score(coords, labels)),
        f"{prefix}_calinski_harabasz": finite_metric_value(calinski_harabasz_score(coords, labels)),
    }


def centroid_separation_metrics(coords: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute centroid and intra-class spread metrics for two labeled groups.

    Args:
        coords (np.ndarray): Two-dimensional embedding coordinates.
        labels (np.ndarray): Integer group labels.

    Returns:
        dict[str, float]: Centroid distance and normalized separation metrics.
    """

    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        return {
            "centroid_distance": float("nan"),
            "mean_intra_group_distance": float("nan"),
            "centroid_distance_over_intra": float("nan"),
        }
    first = coords[labels == unique_labels[0]]
    second = coords[labels == unique_labels[1]]
    first_centroid = np.mean(first, axis=0)
    second_centroid = np.mean(second, axis=0)
    centroid_distance = float(np.linalg.norm(first_centroid - second_centroid))
    first_spread = np.mean(np.linalg.norm(first - first_centroid, axis=1))
    second_spread = np.mean(np.linalg.norm(second - second_centroid, axis=1))
    intra = float((first_spread + second_spread) / 2.0)
    return {
        "centroid_distance": centroid_distance,
        "mean_intra_group_distance": intra,
        "centroid_distance_over_intra": float(centroid_distance / max(intra, 1e-9)),
    }


def label_separation_metrics(coords: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Compute numerical UMAP separation metrics before clustering.

    Args:
        coords (np.ndarray): Two-dimensional UMAP coordinates.
        labels (np.ndarray): Integer group labels for bacteria and particles.

    Returns:
        pd.DataFrame: One-row table of true-label separation metrics.
    """

    metrics = internal_embedding_metrics(coords, labels, prefix="true_label")
    metrics.update(centroid_separation_metrics(coords, labels))
    return pd.DataFrame([metrics])


def cluster_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Compute cluster purity against known bacteria/particle labels.

    Args:
        true_labels (np.ndarray): Ground-truth group labels.
        cluster_labels (np.ndarray): Predicted cluster labels.

    Returns:
        float: Fraction of samples matching the majority true label in their cluster.
    """

    total = len(true_labels)
    if total == 0:
        return float("nan")
    correct = 0
    for cluster_label in np.unique(cluster_labels):
        mask = cluster_labels == cluster_label
        if not np.any(mask):
            continue
        _, counts = np.unique(true_labels[mask], return_counts=True)
        correct += int(np.max(counts))
    return float(correct / total)


def choose_dbscan_eps(coords: np.ndarray, min_samples: int = 10) -> float:
    """Choose a deterministic DBSCAN epsilon from k-nearest-neighbor distances.

    Args:
        coords (np.ndarray): Two-dimensional UMAP coordinates.
        min_samples (int): DBSCAN min_samples value and neighbor rank.

    Returns:
        float: Epsilon distance for DBSCAN.
    """

    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(coords)
    distances, _ = neighbors.kneighbors(coords)
    kth_distances = distances[:, -1]
    return float(np.percentile(kth_distances, 85))


def leiden_cluster_labels(coords: np.ndarray, n_neighbors: int = 20, resolution: float = 0.35) -> np.ndarray:
    """Cluster UMAP coordinates with Leiden on a weighted k-nearest-neighbor graph.

    Args:
        coords (np.ndarray): Two-dimensional UMAP coordinates.
        n_neighbors (int): Number of neighbors used to build the graph.
        resolution (float): Leiden resolution parameter controlling cluster granularity.

    Returns:
        np.ndarray: Leiden cluster labels.
    """

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1)
    neighbors.fit(coords)
    distances, indices = neighbors.kneighbors(coords)
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    seen_edges: set[tuple[int, int]] = set()
    for source_index, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, target_index in zip(row_distances[1:], row_indices[1:]):
            edge = tuple(sorted((int(source_index), int(target_index))))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            edges.append(edge)
            weights.append(float(1.0 / (1.0 + distance)))

    graph = ig.Graph(n=coords.shape[0], edges=edges, directed=False)
    graph.es["weight"] = weights
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
    )
    return np.asarray(partition.membership, dtype=int)


def cluster_umap(coords: np.ndarray) -> dict[str, np.ndarray]:
    """Cluster UMAP coordinates using several standard unsupervised methods.

    Args:
        coords (np.ndarray): Two-dimensional UMAP coordinates.

    Returns:
        dict[str, np.ndarray]: Mapping from method name to cluster labels.
    """

    dbscan_eps = choose_dbscan_eps(coords)
    return {
        "kmeans_5": KMeans(n_clusters=5, random_state=42, n_init=20).fit_predict(coords),
        "gaussian_mixture_5": GaussianMixture(n_components=5, random_state=42).fit_predict(coords),
        "agglomerative_5": AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(coords),
        "spectral_5": SpectralClustering(n_clusters=5, random_state=42, assign_labels="kmeans").fit_predict(coords),
        "dbscan": DBSCAN(eps=dbscan_eps, min_samples=10).fit_predict(coords),
        "leiden": leiden_cluster_labels(coords),
    }


def cluster_quality_metrics(coords: np.ndarray, true_labels: np.ndarray, clusters: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute external and internal quality metrics for each clustering technique.

    Args:
        coords (np.ndarray): Two-dimensional UMAP coordinates.
        true_labels (np.ndarray): Integer bacteria/particle labels.
        clusters (dict[str, np.ndarray]): Cluster labels by method name.

    Returns:
        pd.DataFrame: Per-method clustering quality metrics.
    """

    rows = []
    for method, cluster_labels in clusters.items():
        usable_mask = valid_internal_cluster_labels(cluster_labels)
        usable_coords = coords[usable_mask]
        usable_clusters = cluster_labels[usable_mask]
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, cluster_labels)
        row = {
            "method": method,
            "n_clusters_including_noise": int(len(np.unique(cluster_labels))),
            "n_clusters_excluding_noise": int(len(np.unique(cluster_labels[usable_mask]))) if np.any(usable_mask) else 0,
            "noise_fraction": float(np.mean(cluster_labels == -1)),
            "adjusted_rand_index": finite_metric_value(adjusted_rand_score(true_labels, cluster_labels)),
            "normalized_mutual_info": finite_metric_value(normalized_mutual_info_score(true_labels, cluster_labels)),
            "adjusted_mutual_info": finite_metric_value(adjusted_mutual_info_score(true_labels, cluster_labels)),
            "homogeneity": finite_metric_value(homogeneity),
            "completeness": finite_metric_value(completeness),
            "v_measure": finite_metric_value(v_measure),
            "fowlkes_mallows": finite_metric_value(fowlkes_mallows_score(true_labels, cluster_labels)),
            "purity": cluster_purity(true_labels, cluster_labels),
        }
        row.update(internal_embedding_metrics(usable_coords, usable_clusters, prefix="cluster"))
        rows.append(row)
    return pd.DataFrame(rows)


def image_batch_palette(umap_df: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Build a categorical file-level color palette for batch-effect inspection.

    Args:
        umap_df (pd.DataFrame): UMAP coordinate table with file and group columns.

    Returns:
        dict[str, tuple[float, float, float]]: Mapping from filename to RGB color.
    """

    files = sorted(umap_df["file"].unique())
    colors = sns.color_palette("tab10", n_colors=len(files))
    return dict(zip(files, colors))


def group_markers(umap_df: pd.DataFrame) -> dict[str, str]:
    """Build group-level marker shapes for batch-effect inspection.

    Args:
        umap_df (pd.DataFrame): UMAP coordinate table with group column.

    Returns:
        dict[str, str]: Mapping from group name to matplotlib marker style.
    """

    return {group: GROUP_MARKERS.get(group, "o") for group in sorted(umap_df["group"].unique())}


def plot_feature_scatter(features: pd.DataFrame, output_dir: Path) -> None:
    """Plot interpretable feature scatter views for bacteria/particle separation.

    Args:
        features (pd.DataFrame): Candidate feature table.
        output_dir (Path): Directory where figures are saved.

    Returns:
        None: Figures are written to disk.
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.scatterplot(data=features, x="area", y="sbr", hue="group", alpha=0.35, s=18, ax=axes[0])
    axes[0].set_xscale("log")
    apply_robust_xlim(axes[0], features["area"].to_numpy())
    apply_robust_ylim(axes[0], features["sbr"].to_numpy())
    sns.scatterplot(data=features, x="axis_ratio", y="cnr", hue="group", alpha=0.35, s=18, ax=axes[1])
    apply_robust_xlim(axes[1], features["axis_ratio"].to_numpy())
    apply_robust_ylim(axes[1], features["cnr"].to_numpy())
    sns.scatterplot(data=features, x="eccentricity", y="solidity", hue="group", alpha=0.35, s=18, ax=axes[2])
    apply_robust_xlim(axes[2], features["eccentricity"].to_numpy())
    apply_robust_ylim(axes[2], features["solidity"].to_numpy())
    for ax in axes:
        ax.set_title("")
        ax.set_xlabel(ax.get_xlabel(), fontsize=PANEL_TITLE_FONTSIZE)
        ax.set_ylabel(ax.get_ylabel(), fontsize=PANEL_TITLE_FONTSIZE)
        ax.tick_params(axis="both", labelsize=PANEL_TITLE_FONTSIZE)
        apply_grid(ax)
    save_figure(fig, output_dir / "interpretable_feature_separation.png")


def plot_pca(features: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Plot a PCA projection from standardized interpretable candidate features.

    Args:
        features (pd.DataFrame): Candidate feature table.
        output_dir (Path): Directory where figures are saved.

    Returns:
        pd.DataFrame: PCA coordinates joined with group metadata.
    """

    values = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled = StandardScaler().fit_transform(values)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "group": features["group"].to_numpy(),
            "machine": features["machine"].to_numpy(),
            "file": features["file"].to_numpy(),
        }
    )
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(
        data=pca_df,
        x="pc1",
        y="pc2",
        hue="group",
        alpha=0.35,
        s=22,
        ax=ax,
    )
    ax.set_title(
        "PCA of Candidate Features\n"
        f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.2f}, "
        f"PC2={pca.explained_variance_ratio_[1]:.2f}"
    )
    apply_robust_xlim(ax, pca_df["pc1"].to_numpy())
    apply_robust_ylim(ax, pca_df["pc2"].to_numpy())
    apply_grid(ax)
    save_figure(fig, output_dir / "candidate_feature_pca.png")
    return pca_df


def plot_umap(features: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Plot a UMAP projection from standardized interpretable candidate features.

    Args:
        features (pd.DataFrame): Candidate feature table.
        output_dir (Path): Directory where figures are saved.

    Returns:
        pd.DataFrame: UMAP coordinates joined with group metadata.
    """

    values = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled = StandardScaler().fit_transform(values)
    reducer = UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
    )
    coords = reducer.fit_transform(scaled)
    umap_df = pd.DataFrame(
        {
            "umap1": coords[:, 0],
            "umap2": coords[:, 1],
            "group": features["group"].to_numpy(),
            "machine": features["machine"].to_numpy(),
            "file": features["file"].to_numpy(),
        }
    )
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(
        data=umap_df,
        x="umap1",
        y="umap2",
        hue="group",
        alpha=0.35,
        s=UMAP_DOT_SIZE,
        ax=ax,
    )
    ax.set_title("UMAP of Candidate Features")
    apply_robust_xlim(ax, umap_df["umap1"].to_numpy())
    apply_robust_ylim(ax, umap_df["umap2"].to_numpy())
    apply_grid(ax)
    save_figure(fig, output_dir / "candidate_feature_umap.png")
    return umap_df


def plot_umap_by_image(umap_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot UMAP coordinates colored by source image to inspect batch effects.

    Args:
        umap_df (pd.DataFrame): UMAP coordinate table.
        output_dir (Path): Directory where the figure is saved.

    Returns:
        None: The batch-effect plot is written to disk.
    """

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.scatterplot(
        data=umap_df,
        x="umap1",
        y="umap2",
        hue="file",
        style="group",
        palette=image_batch_palette(umap_df),
        markers=group_markers(umap_df),
        alpha=0.42,
        s=UMAP_DOT_SIZE,
        ax=ax,
    )
    ax.set_title("UMAP by Source Image (Batch Effect Check)")
    apply_robust_xlim(ax, umap_df["umap1"].to_numpy())
    apply_robust_ylim(ax, umap_df["umap2"].to_numpy())
    apply_grid(ax)
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    save_figure(fig, output_dir / "candidate_feature_umap_by_image.png")


def plot_umap_clusters(umap_df: pd.DataFrame, cluster_assignments: pd.DataFrame, output_dir: Path) -> None:
    """Plot UMAP labels, source-image batches, and clustering technique assignments.

    Args:
        umap_df (pd.DataFrame): UMAP coordinate table.
        cluster_assignments (pd.DataFrame): Cluster label table containing one column per method.
        output_dir (Path): Directory where the figure is saved.

    Returns:
        None: The cluster plot is written to disk.
    """

    methods = [column for column in cluster_assignments.columns if column.startswith("cluster_")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    flattened_axes = axes.ravel()

    sns.scatterplot(
        data=umap_df,
        x="umap1",
        y="umap2",
        hue="group",
        alpha=0.38,
        s=UMAP_DOT_SIZE,
        ax=flattened_axes[0],
        legend=False,
    )
    flattened_axes[0].set_title("true labels", fontsize=PANEL_TITLE_FONTSIZE)
    apply_robust_xlim(flattened_axes[0], umap_df["umap1"].to_numpy())
    apply_robust_ylim(flattened_axes[0], umap_df["umap2"].to_numpy())
    apply_grid(flattened_axes[0])

    sns.scatterplot(
        data=umap_df,
        x="umap1",
        y="umap2",
        hue="file",
        style="group",
        palette=image_batch_palette(umap_df),
        markers=group_markers(umap_df),
        alpha=0.42,
        s=UMAP_DOT_SIZE,
        ax=flattened_axes[1],
        legend=False,
    )
    flattened_axes[1].set_title("source image batches", fontsize=PANEL_TITLE_FONTSIZE)
    apply_robust_xlim(flattened_axes[1], umap_df["umap1"].to_numpy())
    apply_robust_ylim(flattened_axes[1], umap_df["umap2"].to_numpy())
    apply_grid(flattened_axes[1])

    for ax, method in zip(flattened_axes[2:], methods):
        plot_df = umap_df.copy()
        plot_df["cluster_label"] = cluster_assignments[method].astype(str).to_numpy()
        sns.scatterplot(
            data=plot_df,
            x="umap1",
            y="umap2",
            hue="cluster_label",
            alpha=0.38,
            s=UMAP_DOT_SIZE,
            ax=ax,
            legend=False,
        )
        ax.set_title(method.replace("cluster_", ""), fontsize=PANEL_TITLE_FONTSIZE)
        apply_robust_xlim(ax, umap_df["umap1"].to_numpy())
        apply_robust_ylim(ax, umap_df["umap2"].to_numpy())
        apply_grid(ax)
    for ax in flattened_axes[: 2 + len(methods)]:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in flattened_axes[2 + len(methods) :]:
        ax.axis("off")
    save_figure(fig, output_dir / "candidate_feature_umap_clusters.png")


def plot_feature_distributions(features: pd.DataFrame, output_dir: Path) -> None:
    """Plot group distributions for the most relevant candidate features.

    Args:
        features (pd.DataFrame): Candidate feature table.
        output_dir (Path): Directory where figures are saved.

    Returns:
        None: Figure is written to disk.
    """

    melted = features.melt(
        id_vars=["group", "file"],
        value_vars=["area", "axis_ratio", "eccentricity", "sbr", "cnr", "solidity"],
        var_name="feature",
        value_name="value",
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.ravel(), ["area", "axis_ratio", "eccentricity", "sbr", "cnr", "solidity"]):
        subset = melted[melted["feature"] == feature]
        sns.violinplot(data=subset, x="group", y="value", ax=ax, cut=0, inner="quartile")
        ax.set_title(feature, fontsize=PANEL_TITLE_FONTSIZE)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if feature in {"area", "axis_ratio"}:
            _, upper = robust_limits(subset["value"].to_numpy())
            ax.set_ylim(0.0, upper)
        elif feature == "cnr":
            ax.set_yscale("symlog")
            apply_robust_ylim(ax, subset["value"].to_numpy())
        else:
            apply_robust_ylim(ax, subset["value"].to_numpy())
        apply_grid(ax)
    save_figure(fig, output_dir / "candidate_feature_distributions.png")


def save_interpretation(
    features: pd.DataFrame,
    summary: pd.DataFrame,
    candidate_counts: pd.DataFrame,
    umap_separation: pd.DataFrame,
    cluster_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write a concise markdown interpretation for presentation use.

    Args:
        features (pd.DataFrame): Candidate feature table.
        summary (pd.DataFrame): Numeric separability summary.
        candidate_counts (pd.DataFrame): Per-image candidate counts.
        umap_separation (pd.DataFrame): True-label UMAP separation metrics.
        cluster_metrics (pd.DataFrame): Cluster quality metrics by method.
        output_dir (Path): Directory where the markdown file is saved.

    Returns:
        None: Markdown interpretation is written to disk.
    """

    top_features = ", ".join(summary.head(5)["feature"].tolist())
    bacteria_candidates = int((features["group"] == "Bacteria").sum())
    particle_candidates = int((features["group"] == "Particles").sum())
    bacteria_count_mean = candidate_counts.loc[candidate_counts["group"] == "Bacteria", "candidate_count"].mean()
    particle_count_mean = candidate_counts.loc[candidate_counts["group"] == "Particles", "candidate_count"].mean()
    umap_silhouette = float(umap_separation.loc[0, "true_label_silhouette"])
    best_cluster = cluster_metrics.sort_values("adjusted_rand_index", ascending=False).iloc[0]
    text = f"""# Separability Evaluation Notes

This script does not train the final discrimination model. It checks whether preprocessing creates cleaner candidate-level evidence for a downstream model.

- Candidate count after preprocessing: {bacteria_candidates} from Set A and {particle_candidates} from Set B.
- Strongest separating features by effect size: {top_features}.
- Mean candidate count per FOV: Set A = {bacteria_count_mean:.1f}, Set B = {particle_count_mean:.1f}.
- UMAP true-label silhouette: {umap_silhouette:.3f}.
- Best UMAP clustering by adjusted Rand index: {best_cluster["method"]} with ARI = {best_cluster["adjusted_rand_index"]:.3f}.
- Interpretation caveat: the dataset has only 10 fields of view and no pixel-level labels, so these numbers are proxy diagnostics rather than validated biological sensitivity or specificity.

For presentation, emphasize that cross-machine harmonization and weak-signal enhancement should be evaluated jointly: the desired behavior is improved Set A local SBR/CNR while tracking how many particle candidates are also admitted by the more permissive settings.
"""
    (output_dir / "separability_interpretation.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run candidate feature extraction and separability visualizations.

    Args:
        None.

    Returns:
        None: Candidate features, PCA coordinates, summaries, and plots are written to disk.
    """

    configure_matplotlib()
    output_dir = ensure_dir(SEPARABILITY_ROOT)
    features = collect_candidate_features()
    features.to_csv(output_dir / "candidate_features.csv", index=False)

    summary = build_separability_summary(features)
    summary.to_csv(output_dir / "feature_separability_summary.csv", index=False)
    candidate_counts = (
        features.groupby(["group", "machine", "file"])
        .size()
        .reset_index(name="candidate_count")
    )
    candidate_counts.to_csv(output_dir / "candidate_count_summary.csv", index=False)

    plot_feature_scatter(features, output_dir)
    pca_df = plot_pca(features, output_dir)
    pca_df.to_csv(output_dir / "candidate_feature_pca_coordinates.csv", index=False)
    umap_df = plot_umap(features, output_dir)
    umap_df.to_csv(output_dir / "candidate_feature_umap_coordinates.csv", index=False)
    true_labels = encoded_group_labels(umap_df["group"])
    umap_separation = label_separation_metrics(umap_df[["umap1", "umap2"]].to_numpy(), true_labels)
    umap_separation.to_csv(output_dir / "umap_true_label_separation_metrics.csv", index=False)
    plot_umap_by_image(umap_df, output_dir)
    clusters = cluster_umap(umap_df[["umap1", "umap2"]].to_numpy())
    cluster_assignments = umap_df[["group", "machine", "file", "umap1", "umap2"]].copy()
    for method, labels in clusters.items():
        cluster_assignments[f"cluster_{method}"] = labels
    cluster_assignments.to_csv(output_dir / "umap_cluster_assignments.csv", index=False)
    cluster_metrics = cluster_quality_metrics(umap_df[["umap1", "umap2"]].to_numpy(), true_labels, clusters)
    cluster_metrics.to_csv(output_dir / "umap_cluster_metrics.csv", index=False)
    plot_umap_clusters(umap_df, cluster_assignments, output_dir)
    plot_feature_distributions(features, output_dir)
    save_interpretation(features, summary, candidate_counts, umap_separation, cluster_metrics, output_dir)
    print(f"Separability evaluation complete: wrote candidate diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
