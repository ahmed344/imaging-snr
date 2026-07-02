"""Exploratory data analysis for all Spore.Bio exercise images."""

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
    baseline_metrics,
    configure_matplotlib,
    ensure_dir,
    estimate_baseline,
    flatten_baseline,
    image_summary,
    load_image_records,
    robust_percentile_values,
    sanitized_stem,
    save_figure,
    show_image,
)


EDA_ROOT = OUTPUT_ROOT / "01_eda"


def plot_image_eda(record_index: int, group_count: int, record_output_dir: Path) -> None:
    """Plot raw image, histogram, profiles, and baseline for one image.

    Args:
        record_index (int): Index of the image in the loaded records list.
        group_count (int): Total number of images, used only for title context.
        record_output_dir (Path): Output directory where the figure will be written.

    Returns:
        None: The plot is saved to disk.
    """

    records = load_image_records()
    record = records[record_index]
    image = record.image
    baseline = estimate_baseline(image)
    flattened = flatten_baseline(image, baseline=baseline)
    row_profile = np.median(image, axis=1)
    col_profile = np.median(image, axis=0)
    flat_row_profile = np.median(flattened, axis=1)
    flat_col_profile = np.median(flattened, axis=0)
    percentiles = robust_percentile_values(image)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    title = f"{record.group} | {record.machine} | {record.path.name} ({record_index + 1}/{group_count})"
    fig.suptitle(title)
    show_image(axes[0, 0], image, "Raw FOV")
    show_image(axes[0, 1], baseline, "Estimated Baseline")
    show_image(axes[0, 2], flattened, "Baseline Flattened")

    histogram_values = np.concatenate([image.ravel(), flattened.ravel()])
    axes[1, 0].hist(image.ravel(), bins=HISTOGRAM_BINS, alpha=0.75, label="raw")
    axes[1, 0].hist(flattened.ravel(), bins=HISTOGRAM_BINS, alpha=0.55, label="flattened")
    for key in ("p1", "p50", "p99"):
        if key in percentiles:
            axes[1, 0].axvline(percentiles[key], linestyle="--", linewidth=1, label=key)
    axes[1, 0].set_title("Intensity Histogram")
    axes[1, 0].set_xlabel("Normalized intensity")
    axes[1, 0].set_ylabel("Pixel count")
    apply_robust_xlim(axes[1, 0], histogram_values)
    apply_grid(axes[1, 0])
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].plot(row_profile, label="raw row median")
    axes[1, 1].plot(flat_row_profile, label="flattened row median")
    axes[1, 1].set_title("Row Median Profile")
    axes[1, 1].set_xlabel("Row")
    axes[1, 1].set_ylabel("Intensity")
    apply_grid(axes[1, 1])
    axes[1, 1].legend(fontsize=7)

    axes[1, 2].plot(col_profile, label="raw column median")
    axes[1, 2].plot(flat_col_profile, label="flattened column median")
    axes[1, 2].set_title("Column Median Profile")
    axes[1, 2].set_xlabel("Column")
    axes[1, 2].set_ylabel("Intensity")
    apply_grid(axes[1, 2])
    axes[1, 2].legend(fontsize=7)

    save_figure(fig, record_output_dir / f"{record.group}_{sanitized_stem(record.path)}_eda.png")


def save_group_histograms(stats: pd.DataFrame, output_dir: Path) -> None:
    """Save grouped histogram and baseline metric summary plots.

    Args:
        stats (pd.DataFrame): Per-image statistics table.
        output_dir (Path): Output directory for summary figures.

    Returns:
        None: Figures are written to disk.
    """

    records = load_image_records()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for record in records:
        linestyle = "--" if record.group == "Particles" else "-"
        sns.kdeplot(
            record.image.ravel()[::20],
            ax=axes[0],
            label=f"{record.group}: {record.path.stem[:16]}",
            alpha=0.6,
            linestyle=linestyle,
        )
    axes[0].set_title("Raw Intensity Distributions (all 10 FOVs)")
    axes[0].set_xlabel("Normalized intensity")
    apply_robust_xlim(axes[0], np.concatenate([record.image.ravel()[::20] for record in records]))
    apply_grid(axes[0])
    axes[0].legend(fontsize=6)

    sns.boxplot(data=stats, x="group", y="baseline_unevenness", ax=axes[1])
    sns.stripplot(data=stats, x="group", y="baseline_unevenness", ax=axes[1], color="black", size=4)
    axes[1].set_title("Low-Frequency Baseline Unevenness")
    axes[1].set_xlabel("Set")
    axes[1].set_ylabel("(p95 - p5) / p50")
    apply_grid(axes[1])
    save_figure(fig, output_dir / "all_images_histograms_and_baseline.png")


def save_montage(
    panels: list[tuple[np.ndarray, str]],
    nrows: int,
    ncols: int,
    output_path: Path,
    figsize: tuple[float, float] = (15, 6),
) -> None:
    """Save a grid montage of images with titles.

    Args:
        panels (list[tuple[np.ndarray, str]]): Image arrays and panel titles.
        nrows (int): Number of montage rows.
        ncols (int): Number of montage columns.
        output_path (Path): Destination file path for the montage figure.
        figsize (tuple[float, float]): Matplotlib figure size in inches.

    Returns:
        None: The montage is written to disk.
    """

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for ax, (image, title) in zip(axes.ravel(), panels):
        show_image(ax, image, title)
    save_figure(fig, output_path)


def save_raw_montage(output_dir: Path) -> None:
    """Save a raw-image montage of all fields of view.

    Args:
        output_dir (Path): Output directory for the montage figure.

    Returns:
        None: The montage is written to disk.
    """

    records = load_image_records()
    panels = [(record.image, f"{record.group}\n{record.path.stem[:24]}") for record in records]
    save_montage(panels, nrows=2, ncols=5, output_path=output_dir / "raw_all_10_fovs.png")


def save_corrected_montage(output_dir: Path) -> None:
    """Save a baseline-flattened montage of all fields of view.

    Args:
        output_dir (Path): Output directory for the montage figure.

    Returns:
        None: The montage is written to disk.
    """

    records = load_image_records()
    panels: list[tuple[np.ndarray, str]] = []
    for record in records:
        baseline = estimate_baseline(record.image)
        flattened = flatten_baseline(record.image, baseline=baseline)
        panels.append((flattened, f"{record.group}\n{record.path.stem[:24]}"))
    save_montage(panels, nrows=2, ncols=5, output_path=output_dir / "corrected_all_10_fovs.png")


def save_group_raw_and_flattened_montage(group: str, output_dir: Path) -> None:
    """Save a two-row montage with raw images on top and flattened images below.

    Args:
        group (str): Image group label, either Bacteria or Particles.
        output_dir (Path): Output directory for the montage figure.

    Returns:
        None: The montage is written to disk.
    """

    records = [record for record in load_image_records() if record.group == group]
    panels: list[tuple[np.ndarray, str]] = []
    for record in records:
        title = f"{record.group}\n{record.path.stem[:24]}"
        panels.append((record.image, title))
    for record in records:
        baseline = estimate_baseline(record.image)
        flattened = flatten_baseline(record.image, baseline=baseline)
        panels.append((flattened, ""))
    filename = f"{group.lower()}_raw_and_flattened.png"
    save_montage(panels, nrows=2, ncols=len(records), output_path=output_dir / filename)


def main() -> None:
    """Run all-image exploratory data analysis.

    Args:
        None.

    Returns:
        None: CSV summaries and figures are written under outputs/01_eda.
    """

    configure_matplotlib()
    output_dir = ensure_dir(EDA_ROOT)
    records = load_image_records()
    rows: list[dict[str, float | str]] = []
    for index, record in enumerate(records):
        baseline = estimate_baseline(record.image)
        flattened = flatten_baseline(record.image, baseline=baseline)
        row = image_summary(record)
        row.update(baseline_metrics(record.image, baseline, flattened))
        rows.append(row)
        plot_image_eda(index, len(records), output_dir)

    stats = pd.DataFrame(rows)
    stats.to_csv(output_dir / "eda_image_statistics.csv", index=False)

    numeric_columns = stats.select_dtypes(include=[np.number]).columns
    group_summary = stats.groupby("group")[numeric_columns].agg(["mean", "min", "max", "std"])
    group_summary.to_csv(output_dir / "eda_group_summary.csv")

    save_raw_montage(output_dir)
    save_corrected_montage(output_dir)
    save_group_raw_and_flattened_montage("Bacteria", output_dir)
    save_group_raw_and_flattened_montage("Particles", output_dir)
    save_group_histograms(stats, output_dir)
    print(f"EDA complete: wrote {len(records)} per-image figures and summaries to {output_dir}")


if __name__ == "__main__":
    main()
