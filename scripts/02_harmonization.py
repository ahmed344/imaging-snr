"""Cross-machine baseline correction and robust intensity harmonization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import (
    HISTOGRAM_BINS,
    OUTPUT_ROOT,
    apply_grid,
    apply_robust_xlim,
    background_residual_std,
    baseline_metrics,
    configure_matplotlib,
    ensure_dir,
    harmonize_record,
    large_artifact_halo_metric,
    load_image_records,
    overlay_mask,
    robust_scale_from_images,
    sanitized_stem,
    save_figure,
    save_image_tiff,
    show_image,
)


HARMONIZATION_ROOT = OUTPUT_ROOT / "02_harmonization"


def display_sample(image: np.ndarray, stride: int = 3) -> np.ndarray:
    """Downsample an image for faster plotting only.

    Args:
        image (np.ndarray): Image or mask to display.
        stride (int): Pixel stride used for display downsampling.

    Returns:
        np.ndarray: Downsampled view of the input image.
    """

    return image[::stride, ::stride]


def save_harmonized_image(output_dir: Path, group: str, source_path: Path, image: np.ndarray) -> Path:
    """Save one harmonized image as TIFF for downstream inspection.

    Args:
        output_dir (Path): Directory where images are stored.
        group (str): Image group name.
        source_path (Path): Original image path.
        image (np.ndarray): Harmonized image to save.

    Returns:
        Path: Saved TIFF image path.
    """

    image_dir = ensure_dir(output_dir / "images" / group)
    return save_image_tiff(image, image_dir / f"{sanitized_stem(source_path)}_harmonized.tif")


def plot_harmonization_stages(
    output_dir: Path,
    group: str,
    file_name: str,
    raw: np.ndarray,
    gaussian_baseline: np.ndarray,
    baseline: np.ndarray,
    foreground_mask: np.ndarray,
    corrected: np.ndarray,
    harmonized: np.ndarray,
) -> None:
    """Plot Gaussian pass, object mask, paraboloid fit, correction, and harmonization.

    Args:
        output_dir (Path): Directory where the plot is saved.
        group (str): Image group label.
        file_name (str): Source filename.
        raw (np.ndarray): Raw image.
        gaussian_baseline (np.ndarray): First-pass Gaussian baseline.
        baseline (np.ndarray): Object-masked paraboloid background.
        foreground_mask (np.ndarray): Segmented foreground excluded from paraboloid fitting.
        corrected (np.ndarray): Paraboloid-corrected image.
        harmonized (np.ndarray): Final harmonized image.

    Returns:
        None: The figure is saved to disk.
    """

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig.suptitle(f"{group} | {file_name}")
    show_image(axes[0, 0], display_sample(raw), "Raw")
    show_image(axes[0, 1], display_sample(gaussian_baseline), "Initial Gaussian Baseline")
    show_image(
        axes[0, 2],
        overlay_mask(display_sample(raw), display_sample(foreground_mask), color=(1.0, 0.2, 0.1)),
        "Excluded Foreground",
        cmap=None,
    )
    show_image(axes[1, 0], display_sample(baseline), "Object-Masked Paraboloid")
    show_image(axes[1, 1], display_sample(corrected), "Paraboloid Corrected")
    show_image(axes[1, 2], display_sample(harmonized), "Machine-Harmonized")

    for image, label_text in ((raw, "raw"), (corrected, "flattened"), (harmonized, "harmonized")):
        axes[2, 0].plot(np.median(image, axis=1), label=label_text)
        axes[2, 1].plot(np.median(image, axis=0), label=label_text)
        axes[2, 2].hist(image.ravel()[::10], bins=HISTOGRAM_BINS, alpha=0.45, label=label_text)
    axes[2, 0].set_title("Row Median Profiles")
    axes[2, 1].set_title("Column Median Profiles")
    axes[2, 2].set_title("Intensity Histogram")
    apply_robust_xlim(axes[2, 2], np.concatenate([raw.ravel()[::10], corrected.ravel()[::10], harmonized.ravel()[::10]]))
    axes[0, 2].text(
        0.02,
        0.98,
        f"Masked pixels: {100.0 * np.mean(foreground_mask):.2f}%",
        color="white",
        va="top",
        ha="left",
        transform=axes[0, 2].transAxes,
    )
    axes[1, 2].text(
        0.0,
        -0.08,
        "Calibration philosophy: dark/flat references should be acquired where possible; "
        "post-acquisition fitting excludes objects so drift and membrane background are corrected "
        "without learning bright artifacts as illumination.",
        va="top",
        transform=axes[1, 2].transAxes,
        wrap=True,
    )
    for ax in axes[2, :3]:
        ax.legend(fontsize=7)
        ax.set_xlabel("Pixel index or intensity")
        apply_grid(ax)
    save_figure(fig, output_dir / f"{group}_{Path(file_name).stem}_harmonization.png")


def plot_harmonization_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Plot before/after harmonization quality metrics.

    Args:
        metrics (pd.DataFrame): Per-image harmonization metrics.
        output_dir (Path): Directory where figures are saved.

    Returns:
        None: The summary figure is written to disk.
    """

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    sns.boxplot(data=metrics, x="group", y="raw_mean", ax=axes[0])
    sns.stripplot(data=metrics, x="group", y="raw_mean", ax=axes[0], color="black", size=4)
    axes[0].set_title("Raw Mean Intensity")
    apply_grid(axes[0])

    sns.boxplot(data=metrics, x="group", y="harmonized_mean", ax=axes[1])
    sns.stripplot(data=metrics, x="group", y="harmonized_mean", ax=axes[1], color="black", size=4)
    axes[1].set_title("Harmonized Mean Intensity")
    apply_grid(axes[1])

    sns.boxplot(data=metrics, x="group", y="paraboloid_background_residual_std", ax=axes[2])
    sns.stripplot(data=metrics, x="group", y="paraboloid_background_residual_std", ax=axes[2], color="black", size=4)
    axes[2].set_title("Background Residual Std")
    apply_grid(axes[2])

    sns.boxplot(data=metrics, x="group", y="halo_reduction_fraction", ax=axes[3])
    sns.stripplot(data=metrics, x="group", y="halo_reduction_fraction", ax=axes[3], color="black", size=4)
    axes[3].set_title("Large-Artifact Halo Reduction")
    apply_grid(axes[3])
    save_figure(fig, output_dir / "harmonization_summary_metrics.png")


def main() -> None:
    """Run baseline correction and cross-machine harmonization.

    Args:
        None.

    Returns:
        None: Harmonized TIFF images, plots, and metrics are written under outputs/02_harmonization.
    """

    configure_matplotlib()
    output_dir = ensure_dir(HARMONIZATION_ROOT)
    records = load_image_records()
    bacteria_records = [record for record in records if record.group == "Bacteria"]
    bacteria_corrected = []
    for record in bacteria_records:
        print(f"Reference correction: {record.path.name}", flush=True)
        preview = harmonize_record(record, reference_low=0.0, reference_high=1.0)
        bacteria_corrected.append(preview.baseline_corrected)
    reference_low, reference_high = robust_scale_from_images(bacteria_corrected, low_pct=5.0, high_pct=99.0)

    rows: list[dict[str, float | str]] = []
    for record in records:
        print(f"Harmonizing: {record.path.name}", flush=True)
        result = harmonize_record(record, reference_low=reference_low, reference_high=reference_high)
        image_path = save_harmonized_image(output_dir, record.group, record.path, result.harmonized)
        gaussian_metrics = baseline_metrics(record.image, result.gaussian_baseline, result.gaussian_corrected)
        paraboloid_metrics = baseline_metrics(record.image, result.baseline, result.baseline_corrected)
        harmonized_baseline = np.full_like(result.harmonized, np.median(result.harmonized))
        harmonized_metrics = baseline_metrics(result.baseline_corrected, harmonized_baseline, result.harmonized)
        gaussian_halo = large_artifact_halo_metric(result.gaussian_corrected, result.foreground_mask)
        paraboloid_halo = large_artifact_halo_metric(result.baseline_corrected, result.foreground_mask)
        rows.append(
            {
                "group": record.group,
                "machine": record.machine,
                "file": record.path.name,
                "image_path": str(image_path),
                "reference_low_p5": result.low_reference,
                "reference_high_p99": result.high_reference,
                "source_low_p5": result.low_source,
                "source_high_p99": result.high_source,
                "raw_mean": gaussian_metrics["raw_mean"],
                "raw_std": gaussian_metrics["raw_std"],
                "gaussian_baseline_unevenness": gaussian_metrics["baseline_unevenness"],
                "paraboloid_baseline_unevenness": paraboloid_metrics["baseline_unevenness"],
                "foreground_mask_fraction": float(np.mean(result.foreground_mask)),
                "gaussian_background_residual_std": background_residual_std(
                    result.gaussian_corrected,
                    result.foreground_mask,
                ),
                "paraboloid_background_residual_std": background_residual_std(
                    result.baseline_corrected,
                    result.foreground_mask,
                ),
                "gaussian_large_artifact_halo": gaussian_halo,
                "paraboloid_large_artifact_halo": paraboloid_halo,
                "halo_reduction_fraction": float((gaussian_halo - paraboloid_halo) / max(gaussian_halo, 1e-9)),
                "flattened_row_profile_std": paraboloid_metrics["corrected_row_profile_std"],
                "flattened_col_profile_std": paraboloid_metrics["corrected_col_profile_std"],
                "harmonized_mean": harmonized_metrics["corrected_mean"],
                "harmonized_std": harmonized_metrics["corrected_std"],
                "harmonized_baseline_unevenness": harmonized_metrics["baseline_unevenness"],
                "harmonized_row_profile_std": harmonized_metrics["corrected_row_profile_std"],
                "harmonized_col_profile_std": harmonized_metrics["corrected_col_profile_std"],
            }
        )
        print(f"Plotting: {record.path.name}", flush=True)
        plot_harmonization_stages(
            output_dir,
            record.group,
            record.path.name,
            record.image,
            result.gaussian_baseline,
            result.baseline,
            result.foreground_mask,
            result.baseline_corrected,
            result.harmonized,
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "harmonization_metrics.csv", index=False)
    numeric_columns = metrics.select_dtypes(include=[np.number]).columns
    metrics.groupby("group")[numeric_columns].agg(["mean", "min", "max", "std"]).to_csv(
        output_dir / "harmonization_group_summary.csv"
    )
    plot_harmonization_summary(metrics, output_dir)
    print(f"Harmonization complete: wrote TIFF images and metrics for {len(records)} images to {output_dir}")


if __name__ == "__main__":
    main()
