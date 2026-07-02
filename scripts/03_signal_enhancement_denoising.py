"""Weak-signal enhancement and denoising for bacteria candidate visibility."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.measure import label

from common import (
    HISTOGRAM_BINS,
    OUTPUT_ROOT,
    apply_grid,
    apply_robust_xlim,
    configure_matplotlib,
    count_local_peaks,
    ensure_dir,
    enhance_signal,
    estimate_noise_mad,
    extract_candidate_features,
    harmonize_record,
    load_image_records,
    overlay_mask,
    robust_scale_from_images,
    sanitized_stem,
    save_figure,
    save_image_tiff,
    show_image,
)


ENHANCEMENT_ROOT = OUTPUT_ROOT / "03_signal_enhancement_denoising"


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


def save_enhancement_images(
    output_dir: Path,
    group: str,
    source_path: Path,
    harmonized: np.ndarray,
    denoised: np.ndarray,
    enhanced: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, str]:
    """Save enhancement stage images as TIFF files for downstream inspection.

    Args:
        output_dir (Path): Directory where images are stored.
        group (str): Image group name.
        source_path (Path): Original image path.
        harmonized (np.ndarray): Harmonized image.
        denoised (np.ndarray): Denoised image.
        enhanced (np.ndarray): Enhanced response image.
        candidate_mask (np.ndarray): Binary candidate mask.

    Returns:
        dict[str, str]: Mapping from stage name to saved TIFF file path.
    """

    image_dir = ensure_dir(output_dir / "images" / group)
    stem = sanitized_stem(source_path)
    paths = {
        "harmonized_path": image_dir / f"{stem}_harmonized.tif",
        "denoised_path": image_dir / f"{stem}_denoised.tif",
        "enhanced_path": image_dir / f"{stem}_enhanced.tif",
        "candidate_mask_path": image_dir / f"{stem}_candidate_mask.tif",
    }
    save_image_tiff(harmonized, paths["harmonized_path"])
    save_image_tiff(denoised, paths["denoised_path"])
    save_image_tiff(enhanced, paths["enhanced_path"])
    save_image_tiff(candidate_mask, paths["candidate_mask_path"])
    return {key: str(value) for key, value in paths.items()}


def plot_enhancement_stages(
    output_dir: Path,
    group: str,
    file_name: str,
    raw: np.ndarray,
    harmonized: np.ndarray,
    denoised: np.ndarray,
    enhanced: np.ndarray,
    candidate_mask: np.ndarray,
) -> None:
    """Plot signal enhancement stages and candidate overlay for one image.

    Args:
        output_dir (Path): Directory where the figure is saved.
        group (str): Image group label.
        file_name (str): Source filename.
        raw (np.ndarray): Raw image.
        harmonized (np.ndarray): Harmonized image.
        denoised (np.ndarray): Denoised image.
        enhanced (np.ndarray): Enhanced response image.
        candidate_mask (np.ndarray): Binary candidate mask.

    Returns:
        None: The figure is saved to disk.
    """

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    fig.suptitle(f"{group} | {file_name}")
    show_image(axes[0, 0], raw, "Raw")
    show_image(axes[0, 1], harmonized, "Harmonized")
    show_image(axes[0, 2], denoised, "Edge-Preserving Denoised")
    show_image(axes[0, 3], enhanced, "Small Bright Response")
    show_image(axes[1, 0], overlay_mask(harmonized, candidate_mask), "Candidates on Harmonized", cmap=None)
    axes[1, 1].hist(harmonized.ravel()[::10], bins=HISTOGRAM_BINS, alpha=0.6, label="harmonized")
    axes[1, 1].hist(denoised.ravel()[::10], bins=HISTOGRAM_BINS, alpha=0.6, label="denoised")
    axes[1, 1].set_title("Denoising Histogram Check")
    apply_robust_xlim(axes[1, 1], np.concatenate([harmonized.ravel()[::10], denoised.ravel()[::10]]))
    apply_grid(axes[1, 1])
    axes[1, 1].legend(fontsize=7)
    axes[1, 2].hist(enhanced.ravel()[::10], bins=HISTOGRAM_BINS, alpha=0.8)
    axes[1, 2].axvline(np.percentile(enhanced, 99.3), color="red", linestyle="--", linewidth=1)
    axes[1, 2].set_title("Enhanced Response Histogram")
    apply_robust_xlim(axes[1, 2], enhanced.ravel()[::10])
    apply_grid(axes[1, 2])
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.0,
        0.9,
        f"Candidate pixels: {int(candidate_mask.sum())}\n"
        f"Candidate objects: {int(label(candidate_mask).max())}\n"
        f"Noise MAD before: {estimate_noise_mad(harmonized):.5f}\n"
        f"Noise MAD after: {estimate_noise_mad(denoised):.5f}",
        va="top",
    )
    save_figure(fig, output_dir / f"{group}_{Path(file_name).stem}_enhancement.png")


def plot_enhancement_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Save summary plots for enhancement quality metrics.

    Args:
        metrics (pd.DataFrame): Per-image enhancement metrics.
        output_dir (Path): Output directory for summary figures.

    Returns:
        None: The figure is written to disk.
    """

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for axis, metric, title in (
        (axes[0], "noise_reduction_fraction", "Noise Reduction"),
        (axes[1], "candidate_count", "Candidate Count"),
        (axes[2], "median_candidate_sbr", "Median Candidate SBR"),
        (axes[3], "median_candidate_cnr", "Median Candidate CNR"),
    ):
        sns.boxplot(data=metrics, x="group", y=metric, ax=axis)
        sns.stripplot(data=metrics, x="group", y=metric, ax=axis, color="black", size=4)
        axis.set_title(title)
        axis.set_xlabel("Set")
        apply_grid(axis)
    save_figure(fig, output_dir / "enhancement_summary_metrics.png")


def main() -> None:
    """Run denoising, small-object enhancement, and candidate proxy evaluation.

    Args:
        None.

    Returns:
        None: Stage TIFF images, candidate features, metrics, and plots are written to disk.
    """

    configure_matplotlib()
    output_dir = ensure_dir(ENHANCEMENT_ROOT)
    reference_low, reference_high = compute_reference_anchors()
    records = load_image_records()
    metric_rows: list[dict[str, float | str]] = []
    feature_rows: list[dict[str, float | str]] = []

    for record in records:
        harmonized_result = harmonize_record(record, reference_low=reference_low, reference_high=reference_high)
        denoised, enhanced = enhance_signal(harmonized_result.harmonized)
        features, candidate_mask = extract_candidate_features(
            enhanced,
            harmonized_result.harmonized,
            record.path.name,
            record.group,
            record.machine,
        )
        feature_rows.extend(features)
        paths = save_enhancement_images(
            output_dir,
            record.group,
            record.path,
            harmonized_result.harmonized,
            denoised,
            enhanced,
            candidate_mask,
        )
        candidate_sbr = [float(row["sbr"]) for row in features]
        candidate_cnr = [float(row["cnr"]) for row in features]
        before_noise = estimate_noise_mad(harmonized_result.harmonized)
        after_noise = estimate_noise_mad(denoised)
        metric_rows.append(
            {
                "group": record.group,
                "machine": record.machine,
                "file": record.path.name,
                "noise_mad_before": before_noise,
                "noise_mad_after": after_noise,
                "noise_reduction_fraction": float((before_noise - after_noise) / max(before_noise, 1e-9)),
                "enhanced_p99_3": float(np.percentile(enhanced, 99.3)),
                "candidate_count": len(features),
                "candidate_pixels": int(candidate_mask.sum()),
                "local_peak_count": count_local_peaks(enhanced),
                "median_candidate_sbr": float(np.median(candidate_sbr)) if candidate_sbr else 0.0,
                "median_candidate_cnr": float(np.median(candidate_cnr)) if candidate_cnr else 0.0,
                "mean_candidate_area": float(np.mean([row["area"] for row in features])) if features else 0.0,
                **paths,
            }
        )
        plot_enhancement_stages(
            output_dir,
            record.group,
            record.path.name,
            record.image,
            harmonized_result.harmonized,
            denoised,
            enhanced,
            candidate_mask,
        )

    metrics = pd.DataFrame(metric_rows)
    features = pd.DataFrame(feature_rows)
    metrics.to_csv(output_dir / "enhancement_metrics.csv", index=False)
    features.to_csv(output_dir / "candidate_features_from_enhancement.csv", index=False)
    numeric_columns = metrics.select_dtypes(include=[np.number]).columns
    metrics.groupby("group")[numeric_columns].agg(["mean", "min", "max", "std"]).to_csv(
        output_dir / "enhancement_group_summary.csv"
    )
    plot_enhancement_summary(metrics, output_dir)
    print(f"Enhancement complete: wrote metrics and candidate features to {output_dir}")


if __name__ == "__main__":
    main()
