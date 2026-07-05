"""Shared utilities for the Spore.Bio Part A weak-signal pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from scipy import ndimage as ndi
from skimage import exposure, img_as_float, io
from skimage.color import gray2rgb
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, white_tophat
from skimage.restoration import denoise_bilateral
from tifffile import imwrite


DATA_ROOT = Path("data_sporebio")
OUTPUT_ROOT = Path("outputs")
GROUP_TO_MACHINE = {"Bacteria": "Machine 1", "Particles": "Machine 2"}
FIGURE_EXT = ".png"
TIFF_IMAGE_EXT = ".tif"
INTENSITY_DENSITY_SUBSAMPLE = 10


@dataclass(frozen=True)
class ImageRecord:
    """Container for one raw field of view."""

    group: str
    machine: str
    path: Path
    image: np.ndarray


@dataclass(frozen=True)
class HarmonizationResult:
    """Container for harmonization intermediate images and metrics."""

    gaussian_baseline: np.ndarray
    gaussian_corrected: np.ndarray
    baseline: np.ndarray
    foreground_mask: np.ndarray
    baseline_corrected: np.ndarray
    harmonized: np.ndarray
    low_reference: float
    high_reference: float
    low_source: float
    high_source: float


@dataclass(frozen=True)
class EnhancementResult:
    """Container for enhancement intermediate images."""

    baseline: np.ndarray
    baseline_corrected: np.ndarray
    harmonized: np.ndarray
    denoised: np.ndarray
    enhanced: np.ndarray
    candidate_mask: np.ndarray


@dataclass(frozen=True)
class BackgroundCorrectionResult:
    """Container for object-masked background correction outputs."""

    gaussian_baseline: np.ndarray
    gaussian_corrected: np.ndarray
    foreground_mask: np.ndarray
    paraboloid_baseline: np.ndarray
    corrected: np.ndarray
    coefficients: np.ndarray


def configure_matplotlib() -> None:
    """Configure plotting defaults.

    Args:
        None.

    Returns:
        None: This function updates matplotlib runtime configuration in-place.
    """

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.grid": False,
            "image.cmap": "gray",
        }
    )


def robust_limits(values: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> tuple[float, float]:
    """Compute robust plotting limits that are not dominated by outliers.

    Args:
        values (np.ndarray): Numeric values to summarize.
        low_pct (float): Lower percentile for the axis limit.
        high_pct (float): Upper percentile for the axis limit.

    Returns:
        tuple[float, float]: Lower and upper plotting limits.
    """

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [low_pct, high_pct])
    if np.isclose(low, high):
        padding = max(abs(float(low)) * 0.05, 1e-3)
        return float(low - padding), float(high + padding)
    padding = 0.03 * float(high - low)
    return float(low - padding), float(high + padding)


def apply_grid(ax: Axes) -> None:
    """Apply a light grid to a non-image plot.

    Args:
        ax (Axes): Matplotlib axes to update.

    Returns:
        None: The axes grid is updated in-place.
    """

    ax.grid(True, alpha=0.3)


def apply_robust_xlim(ax: Axes, values: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> None:
    """Apply robust x-axis limits to avoid outlier-dominated plots.

    Args:
        ax (Axes): Matplotlib axes to update.
        values (np.ndarray): Values used to compute x-axis limits.
        low_pct (float): Lower percentile for the axis limit.
        high_pct (float): Upper percentile for the axis limit.

    Returns:
        None: The x-axis limits are updated in-place.
    """

    if ax.get_xscale() == "log":
        positive_values = np.asarray(values, dtype=float)
        positive_values = positive_values[np.isfinite(positive_values) & (positive_values > 0.0)]
        if positive_values.size == 0:
            return
        low, high = robust_limits(positive_values, low_pct=low_pct, high_pct=high_pct)
        ax.set_xlim(max(low, np.min(positive_values) * 0.9), high)
        return
    ax.set_xlim(*robust_limits(values, low_pct=low_pct, high_pct=high_pct))


def apply_robust_ylim(ax: Axes, values: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> None:
    """Apply robust y-axis limits to avoid outlier-dominated plots.

    Args:
        ax (Axes): Matplotlib axes to update.
        values (np.ndarray): Values used to compute y-axis limits.
        low_pct (float): Lower percentile for the axis limit.
        high_pct (float): Upper percentile for the axis limit.

    Returns:
        None: The y-axis limits are updated in-place.
    """

    ax.set_ylim(*robust_limits(values, low_pct=low_pct, high_pct=high_pct))


def plot_intensity_density(
    ax: Axes,
    values: np.ndarray,
    label: str | None = None,
    subsample: int = INTENSITY_DENSITY_SUBSAMPLE,
    **kwargs,
) -> None:
    """Plot a normalized intensity density curve using Gaussian KDE.

    Args:
        ax (Axes): Matplotlib axes to draw into.
        values (np.ndarray): Intensity samples.
        label (str | None): Optional legend label for the curve.
        subsample (int): Pixel stride used before KDE for performance.
        **kwargs: Additional keyword arguments forwarded to sns.kdeplot.

    Returns:
        None: The density curve is drawn into the provided axes.
    """

    plot_kwargs = dict(kwargs)
    if label is not None:
        plot_kwargs["label"] = label
    sns.kdeplot(values.ravel()[::subsample], ax=ax, **plot_kwargs)


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist.

    Args:
        path (Path): Directory path to create.

    Returns:
        Path: The same directory path after creation.
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


def load_image_records(data_root: Path = DATA_ROOT) -> list[ImageRecord]:
    """Load every JPEG image in the expected Spore.Bio exercise layout.

    Args:
        data_root (Path): Root directory containing the Bacteria and Particles folders.

    Returns:
        list[ImageRecord]: Loaded grayscale images with group and machine metadata.
    """

    records: list[ImageRecord] = []
    for group in ("Bacteria", "Particles"):
        group_dir = data_root / group
        for path in sorted(group_dir.glob("*.jpg")):
            records.append(
                ImageRecord(
                    group=group,
                    machine=GROUP_TO_MACHINE[group],
                    path=path,
                    image=read_gray_image(path),
                )
            )
    if not records:
        raise FileNotFoundError(f"No JPEG images found under {data_root}")
    return records


def read_gray_image(path: Path) -> np.ndarray:
    """Read an image as floating-point grayscale in the range [0, 1].

    Args:
        path (Path): Image file path.

    Returns:
        np.ndarray: Two-dimensional grayscale image with dtype float64.
    """

    image = io.imread(path, as_gray=True)
    return img_as_float(image)


def robust_percentile_values(
    image: np.ndarray, percentiles: Sequence[float] = (1, 5, 50, 95, 99, 99.9)
) -> dict[str, float]:
    """Compute named robust percentile values for an image.

    Args:
        image (np.ndarray): Input image.
        percentiles (Sequence[float]): Percentiles to compute.

    Returns:
        dict[str, float]: Mapping from percentile label to percentile value.
    """

    values = np.percentile(image, percentiles)
    return {f"p{str(p).replace('.', '_')}": float(v) for p, v in zip(percentiles, values)}


def image_summary(record: ImageRecord) -> dict[str, float | str]:
    """Summarize one raw image with intensity and shape statistics.

    Args:
        record (ImageRecord): Loaded image record.

    Returns:
        dict[str, float | str]: Per-image summary statistics.
    """

    image = record.image
    percentiles = robust_percentile_values(image)
    summary: dict[str, float | str] = {
        "group": record.group,
        "machine": record.machine,
        "file": record.path.name,
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "saturated_low_fraction": float(np.mean(image <= 0.0)),
        "saturated_high_fraction": float(np.mean(image >= 1.0)),
    }
    summary.update(percentiles)
    return summary


def estimate_baseline(image: np.ndarray, sigma: float = 60.0) -> np.ndarray:
    """Estimate the low-frequency background and illumination field.

    Args:
        image (np.ndarray): Input grayscale image.
        sigma (float): Gaussian sigma in pixels for low-pass baseline estimation.

    Returns:
        np.ndarray: Smooth baseline image with the same shape as the input.
    """

    return gaussian(image, sigma=sigma, preserve_range=True, mode="reflect")


def flatten_baseline(
    image: np.ndarray, baseline: np.ndarray | None = None, sigma: float = 60.0
) -> np.ndarray:
    """Remove low-frequency baseline while preserving the global median intensity.

    Args:
        image (np.ndarray): Input grayscale image.
        baseline (np.ndarray | None): Optional precomputed baseline field.
        sigma (float): Gaussian sigma used when baseline is not provided.

    Returns:
        np.ndarray: Baseline-flattened image clipped to [0, 1].
    """

    estimated = estimate_baseline(image, sigma=sigma) if baseline is None else baseline
    flattened = image - estimated + float(np.median(estimated))
    return np.clip(flattened, 0.0, 1.0)


def robust_mad_threshold(image: np.ndarray, sigma_multiplier: float = 4.0) -> float:
    """Compute a median absolute deviation threshold.

    Args:
        image (np.ndarray): Input response image.
        sigma_multiplier (float): Robust sigma multiplier above the median.

    Returns:
        float: Threshold value in the input image units.
    """

    median = float(np.median(image))
    mad = float(np.median(np.abs(image - median)))
    robust_sigma = 1.4826 * mad
    return median + sigma_multiplier * robust_sigma


def segment_foreground_for_background_fit(
    image: np.ndarray,
    gaussian_corrected: np.ndarray,
    dilation_radius: int = 12,
    min_area: int = 3,
) -> np.ndarray:
    """Segment foreground structures to exclude from background fitting.

    Args:
        image (np.ndarray): Raw grayscale image.
        gaussian_corrected (np.ndarray): First-pass Gaussian baseline corrected image.
        dilation_radius (int): Pixel radius used to dilate foreground regions.
        min_area (int): Minimum connected-component area retained before dilation.

    Returns:
        np.ndarray: Boolean mask where True marks bacteria, particles, fibres, or artifacts.
    """

    local_background = gaussian(gaussian_corrected, sigma=4.0, preserve_range=True, mode="reflect")
    positive_residual = gaussian_corrected - local_background
    raw_high = image > np.percentile(image, 99.4)
    corrected_high = gaussian_corrected > np.percentile(gaussian_corrected, 99.2)
    residual_high = positive_residual > robust_mad_threshold(positive_residual, sigma_multiplier=4.0)
    mask = raw_high | corrected_high | residual_high

    labeled = label(mask)
    cleaned = np.zeros_like(mask, dtype=bool)
    for region in regionprops(labeled):
        if region.area >= min_area:
            cleaned[labeled == region.label] = True

    dilated = ndi.binary_dilation(
        cleaned,
        structure=np.ones((3, 3), dtype=bool),
        iterations=dilation_radius,
    )
    return ndi.binary_fill_holes(dilated)


def paraboloid_design_matrix(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Build a 2D quadratic design matrix with normalized coordinates.

    Args:
        rows (np.ndarray): Row coordinates.
        cols (np.ndarray): Column coordinates.
        shape (tuple[int, int]): Image shape as height and width.

    Returns:
        np.ndarray: Matrix with columns x^2, y^2, xy, x, y, and constant.
    """

    height, width = shape
    y = (rows.astype(float) / max(height - 1, 1)) * 2.0 - 1.0
    x = (cols.astype(float) / max(width - 1, 1)) * 2.0 - 1.0
    return np.column_stack((x * x, y * y, x * y, x, y, np.ones_like(x)))


def evaluate_paraboloid(coefficients: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Evaluate a fitted 2D paraboloid on a full image grid.

    Args:
        coefficients (np.ndarray): Quadratic coefficients ordered as x^2, y^2, xy, x, y, constant.
        shape (tuple[int, int]): Output image shape as height and width.

    Returns:
        np.ndarray: Fitted paraboloid image.
    """

    height, width = shape
    y = np.linspace(-1.0, 1.0, height, dtype=float)[:, np.newaxis]
    x = np.linspace(-1.0, 1.0, width, dtype=float)[np.newaxis, :]
    return (
        coefficients[0] * x * x
        + coefficients[1] * y * y
        + coefficients[2] * x * y
        + coefficients[3] * x
        + coefficients[4] * y
        + coefficients[5]
    )


def gaussian_background_fallback(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a safe Gaussian fallback background and constant paraboloid coefficients.

    Args:
        image (np.ndarray): Raw grayscale image.

    Returns:
        tuple[np.ndarray, np.ndarray]: Gaussian fallback background and coefficient vector.
    """

    fallback = estimate_baseline(image)
    coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0, float(np.median(fallback))])
    return fallback, coefficients


def fit_paraboloid_background(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    sample_stride: int = 12,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2D paraboloid background using only unmasked background pixels.

    Args:
        image (np.ndarray): Raw grayscale image to model.
        foreground_mask (np.ndarray): Boolean mask where True pixels are excluded.
        sample_stride (int): Pixel stride for fitting to keep least-squares lightweight.
        low_pct (float): Low percentile for rejecting residual dark outliers.
        high_pct (float): High percentile for rejecting residual bright outliers.

    Returns:
        tuple[np.ndarray, np.ndarray]: Fitted background image and six paraboloid coefficients.
    """

    background_mask = (~foreground_mask) & np.isfinite(image)
    sampled = np.zeros_like(background_mask, dtype=bool)
    sampled[::sample_stride, ::sample_stride] = True
    fit_mask = background_mask & sampled
    if int(np.sum(fit_mask)) < 500:
        return gaussian_background_fallback(image)

    values = image[fit_mask]
    low, high = np.percentile(values, [low_pct, high_pct])
    fit_mask &= (image >= low) & (image <= high)
    if int(np.sum(fit_mask)) < 500:
        return gaussian_background_fallback(image)
    rows, cols = np.nonzero(fit_mask)
    values = image[fit_mask]
    design = paraboloid_design_matrix(rows, cols, image.shape)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    fitted = evaluate_paraboloid(coefficients, image.shape)
    return np.clip(fitted, 0.0, 1.0), coefficients


def object_masked_background_correction(
    image: np.ndarray,
    sigma: float = 60.0,
    dilation_radius: int = 12,
) -> BackgroundCorrectionResult:
    """Correct smooth background using object-masked paraboloid fitting.

    Args:
        image (np.ndarray): Raw grayscale image.
        sigma (float): Gaussian sigma for the first-pass baseline correction.
        dilation_radius (int): Pixel radius for dilating segmented foreground before fitting.

    Returns:
        BackgroundCorrectionResult: Gaussian pass, foreground mask, paraboloid fit, and corrected image.
    """

    gaussian_baseline = estimate_baseline(image, sigma=sigma)
    gaussian_corrected = flatten_baseline(image, baseline=gaussian_baseline)
    foreground_mask = segment_foreground_for_background_fit(
        image,
        gaussian_corrected,
        dilation_radius=dilation_radius,
    )
    paraboloid_baseline, coefficients = fit_paraboloid_background(image, foreground_mask)
    corrected = image - paraboloid_baseline + float(np.median(paraboloid_baseline[~foreground_mask]))
    corrected = np.clip(corrected, 0.0, 1.0)
    return BackgroundCorrectionResult(
        gaussian_baseline=gaussian_baseline,
        gaussian_corrected=gaussian_corrected,
        foreground_mask=foreground_mask,
        paraboloid_baseline=paraboloid_baseline,
        corrected=corrected,
        coefficients=coefficients,
    )


def background_residual_std(corrected: np.ndarray, foreground_mask: np.ndarray) -> float:
    """Measure background-only residual variation after correction.

    Args:
        corrected (np.ndarray): Background-corrected image.
        foreground_mask (np.ndarray): Boolean foreground mask excluded from the metric.

    Returns:
        float: Standard deviation over background pixels only.
    """

    background = corrected[~foreground_mask]
    if background.size == 0:
        return float(np.std(corrected))
    return float(np.std(background))


def large_artifact_halo_metric(
    corrected: np.ndarray,
    foreground_mask: np.ndarray,
    min_area: int = 300,
    annulus_radius: int = 24,
) -> float:
    """Estimate dark or bright halo magnitude around large masked artifacts.

    Args:
        corrected (np.ndarray): Corrected image to evaluate.
        foreground_mask (np.ndarray): Boolean foreground mask containing candidate artifacts.
        min_area (int): Minimum component area treated as a large artifact.
        annulus_radius (int): Pixel radius used to form the local annulus.

    Returns:
        float: Median absolute local annulus deviation from global background median.
    """

    background_median = float(np.median(corrected[~foreground_mask])) if np.any(~foreground_mask) else float(np.median(corrected))
    labeled = label(foreground_mask)
    halo_values: list[float] = []
    for region in regionprops(labeled):
        if region.area < min_area:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        row_start = max(min_row - annulus_radius, 0)
        row_end = min(max_row + annulus_radius, corrected.shape[0])
        col_start = max(min_col - annulus_radius, 0)
        col_end = min(max_col + annulus_radius, corrected.shape[1])
        local_slice = np.s_[row_start:row_end, col_start:col_end]
        local_component = labeled[local_slice] == region.label
        local_foreground = foreground_mask[local_slice]
        annulus = (
            ndi.binary_dilation(
                local_component,
                structure=np.ones((3, 3), dtype=bool),
                iterations=annulus_radius,
            )
            & ~local_foreground
        )
        if np.any(annulus):
            halo_values.append(abs(float(np.median(corrected[local_slice][annulus])) - background_median))
    if not halo_values:
        return 0.0
    return float(np.median(halo_values))


def baseline_metrics(image: np.ndarray, baseline: np.ndarray, corrected: np.ndarray) -> dict[str, float]:
    """Measure illumination unevenness and residual profile variation.

    Args:
        image (np.ndarray): Raw input image.
        baseline (np.ndarray): Estimated low-frequency baseline.
        corrected (np.ndarray): Corrected image after baseline flattening or harmonization.

    Returns:
        dict[str, float]: Baseline and profile quality metrics.
    """

    baseline_p5, baseline_p50, baseline_p95 = np.percentile(baseline, [5, 50, 95])
    row_profile = np.median(corrected, axis=1)
    col_profile = np.median(corrected, axis=0)
    return {
        "raw_mean": float(np.mean(image)),
        "raw_std": float(np.std(image)),
        "baseline_p5": float(baseline_p5),
        "baseline_p50": float(baseline_p50),
        "baseline_p95": float(baseline_p95),
        "baseline_unevenness": float((baseline_p95 - baseline_p5) / max(abs(baseline_p50), 1e-9)),
        "corrected_row_profile_std": float(np.std(row_profile)),
        "corrected_col_profile_std": float(np.std(col_profile)),
        "corrected_mean": float(np.mean(corrected)),
        "corrected_std": float(np.std(corrected)),
    }


def robust_scale_from_images(
    images: Iterable[np.ndarray], low_pct: float = 5.0, high_pct: float = 99.0
) -> tuple[float, float]:
    """Estimate robust low and high intensity anchors from a collection of images.

    Args:
        images (Iterable[np.ndarray]): Images used to estimate the anchors.
        low_pct (float): Low percentile anchor.
        high_pct (float): High percentile anchor.

    Returns:
        tuple[float, float]: Low and high robust intensity anchors.
    """

    lows: list[float] = []
    highs: list[float] = []
    for image in images:
        low, high = np.percentile(image, [low_pct, high_pct])
        lows.append(float(low))
        highs.append(float(high))
    return float(np.median(lows)), float(np.median(highs))


def match_intensity_anchors(
    image: np.ndarray,
    source_low: float,
    source_high: float,
    reference_low: float,
    reference_high: float,
) -> np.ndarray:
    """Linearly match robust source anchors to robust reference anchors.

    Args:
        image (np.ndarray): Input image to transform.
        source_low (float): Low anchor measured in the source image or group.
        source_high (float): High anchor measured in the source image or group.
        reference_low (float): Desired low anchor in the reference domain.
        reference_high (float): Desired high anchor in the reference domain.

    Returns:
        np.ndarray: Anchor-matched image clipped to [0, 1].
    """

    source_span = max(source_high - source_low, 1e-9)
    reference_span = max(reference_high - reference_low, 1e-9)
    matched = (image - source_low) / source_span
    matched = matched * reference_span + reference_low
    return np.clip(matched, 0.0, 1.0)


def harmonize_record(
    record: ImageRecord,
    reference_low: float,
    reference_high: float,
    sigma: float = 60.0,
    low_pct: float = 5.0,
    high_pct: float = 99.0,
) -> HarmonizationResult:
    """Object-mask background-correct and robustly match one image to the reference domain.

    Args:
        record (ImageRecord): Image record to harmonize.
        reference_low (float): Reference low intensity anchor.
        reference_high (float): Reference high intensity anchor.
        sigma (float): Gaussian sigma for first-pass low-frequency baseline estimation.
        low_pct (float): Source low percentile anchor.
        high_pct (float): Source high percentile anchor.

    Returns:
        HarmonizationResult: Intermediate and final harmonization outputs.
    """

    background_result = object_masked_background_correction(record.image, sigma=sigma)
    corrected = background_result.corrected
    source_low, source_high = np.percentile(corrected, [low_pct, high_pct])
    harmonized = match_intensity_anchors(
        corrected,
        float(source_low),
        float(source_high),
        reference_low,
        reference_high,
    )
    return HarmonizationResult(
        gaussian_baseline=background_result.gaussian_baseline,
        gaussian_corrected=background_result.gaussian_corrected,
        baseline=background_result.paraboloid_baseline,
        foreground_mask=background_result.foreground_mask,
        baseline_corrected=corrected,
        harmonized=harmonized,
        low_reference=reference_low,
        high_reference=reference_high,
        low_source=float(source_low),
        high_source=float(source_high),
    )


def estimate_noise_mad(image: np.ndarray) -> float:
    """Estimate high-frequency noise using a median absolute deviation residual.

    Args:
        image (np.ndarray): Input grayscale image.

    Returns:
        float: Robust noise estimate in image intensity units.
    """

    residual = image - gaussian(image, sigma=2.0, preserve_range=True, mode="reflect")
    mad = np.median(np.abs(residual - np.median(residual)))
    return float(1.4826 * mad)


def enhance_signal(
    harmonized: np.ndarray,
    tophat_radius: int = 5,
    denoise_sigma_color: float = 0.035,
    denoise_sigma_spatial: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Denoise and enhance bright small-scale candidate structures.

    Args:
        harmonized (np.ndarray): Baseline-corrected and machine-harmonized image.
        tophat_radius (int): Disk radius for white top-hat enhancement in pixels.
        denoise_sigma_color (float): Bilateral filter intensity sigma.
        denoise_sigma_spatial (float): Bilateral filter spatial sigma in pixels.

    Returns:
        tuple[np.ndarray, np.ndarray]: Denoised image and enhanced response image.
    """

    denoised = denoise_bilateral(
        harmonized,
        sigma_color=denoise_sigma_color,
        sigma_spatial=denoise_sigma_spatial,
        channel_axis=None,
    )
    small_bright = white_tophat(denoised, footprint=disk(tophat_radius))
    bandpass = gaussian(denoised, sigma=1.0, preserve_range=True) - gaussian(
        denoised, sigma=4.0, preserve_range=True
    )
    enhanced = np.clip(0.65 * exposure.rescale_intensity(small_bright) + 0.35 * exposure.rescale_intensity(bandpass), 0.0, 1.0)
    return denoised, enhanced


def candidate_threshold(enhanced: np.ndarray, percentile: float = 99.3, noise_multiplier: float = 3.0) -> float:
    """Compute a permissive threshold for weak-signal candidate extraction.

    Args:
        enhanced (np.ndarray): Enhanced response image.
        percentile (float): High percentile floor for thresholding.
        noise_multiplier (float): Robust noise multiplier above the median.

    Returns:
        float: Candidate threshold.
    """

    robust_noise = estimate_noise_mad(enhanced)
    median_floor = float(np.median(enhanced) + noise_multiplier * robust_noise)
    percentile_floor = float(np.percentile(enhanced, percentile))
    try:
        otsu_floor = float(threshold_otsu(enhanced))
    except ValueError:
        otsu_floor = 0.0
    otsu_or_percentile = min(percentile_floor, otsu_floor) if otsu_floor > 0.0 else percentile_floor
    return max(median_floor, otsu_or_percentile)


def candidate_mask_from_enhanced(
    enhanced: np.ndarray,
    min_area: int = 2,
    max_area: int = 900,
    percentile: float = 99.3,
) -> np.ndarray:
    """Extract a binary mask of bright candidate objects from an enhanced image.

    Args:
        enhanced (np.ndarray): Enhanced response image.
        min_area (int): Minimum candidate area in pixels.
        max_area (int): Maximum candidate area in pixels.
        percentile (float): High-percentile threshold floor.

    Returns:
        np.ndarray: Boolean candidate mask.
    """

    threshold = candidate_threshold(enhanced, percentile=percentile)
    mask = enhanced > threshold
    labeled = label(mask)
    cleaned = np.zeros_like(mask, dtype=bool)
    for region in regionprops(labeled):
        if min_area <= region.area <= max_area:
            cleaned[labeled == region.label] = True
    return cleaned


def local_background_stats(
    image: np.ndarray, region_mask: np.ndarray, dilation_radius: int = 8
) -> tuple[float, float]:
    """Estimate local background around a candidate region.

    Args:
        image (np.ndarray): Intensity image used for local measurements.
        region_mask (np.ndarray): Boolean mask for one candidate region.
        dilation_radius (int): Annulus dilation radius in pixels.

    Returns:
        tuple[float, float]: Local background mean and standard deviation.
    """

    dilated = ndi.binary_dilation(region_mask, structure=disk(dilation_radius))
    annulus = dilated & ~region_mask
    if not np.any(annulus):
        return float(np.median(image)), float(np.std(image))
    values = image[annulus]
    return float(np.mean(values)), float(np.std(values) + 1e-9)


def local_background_stats_for_region(
    image: np.ndarray,
    labeled: np.ndarray,
    label_id: int,
    bbox: tuple[int, int, int, int],
    dilation_radius: int = 8,
) -> tuple[float, float]:
    """Estimate local background with a cropped annulus around one labeled region.

    Args:
        image (np.ndarray): Intensity image used for local measurements.
        labeled (np.ndarray): Labeled candidate mask.
        label_id (int): Label value for the candidate region.
        bbox (tuple[int, int, int, int]): Region bounding box as min row, min col, max row, max col.
        dilation_radius (int): Annulus dilation radius in pixels.

    Returns:
        tuple[float, float]: Local background mean and standard deviation.
    """

    min_row, min_col, max_row, max_col = bbox
    row_start = max(min_row - dilation_radius, 0)
    row_end = min(max_row + dilation_radius, image.shape[0])
    col_start = max(min_col - dilation_radius, 0)
    col_end = min(max_col + dilation_radius, image.shape[1])
    local_slice = np.s_[row_start:row_end, col_start:col_end]
    local_region = labeled[local_slice] == label_id
    local_image = image[local_slice]
    dilated = ndi.binary_dilation(local_region, structure=disk(dilation_radius))
    annulus = dilated & ~local_region
    if not np.any(annulus):
        return float(np.median(local_image)), float(np.std(local_image) + 1e-9)
    values = local_image[annulus]
    return float(np.mean(values)), float(np.std(values) + 1e-9)


def region_float_property(region: object, preferred_name: str, fallback_name: str) -> float:
    """Read a numeric region property across scikit-image naming versions.

    Args:
        region (object): RegionProperties-like object.
        preferred_name (str): Preferred modern property name.
        fallback_name (str): Legacy property name.

    Returns:
        float: Region property value.
    """

    try:
        return float(getattr(region, preferred_name))
    except AttributeError:
        return float(getattr(region, fallback_name))


def extract_candidate_features(
    enhanced: np.ndarray,
    intensity_image: np.ndarray,
    source_file: str,
    group: str,
    machine: str,
    min_area: int = 2,
    max_area: int = 900,
    max_candidates: int = 5000,
) -> tuple[list[dict[str, float | str]], np.ndarray]:
    """Extract object-level morphology, intensity, and local background features.

    Args:
        enhanced (np.ndarray): Enhanced response image for segmentation.
        intensity_image (np.ndarray): Harmonized intensity image for measurements.
        source_file (str): Source image filename.
        group (str): Biological or interference set label.
        machine (str): Machine label.
        min_area (int): Minimum candidate area in pixels.
        max_area (int): Maximum candidate area in pixels.
        max_candidates (int): Maximum number of strongest candidates to return per image.

    Returns:
        tuple[list[dict[str, float | str]], np.ndarray]: Candidate feature rows and binary mask.
    """

    mask = candidate_mask_from_enhanced(enhanced, min_area=min_area, max_area=max_area)
    labeled = label(mask)
    regions = sorted(
        regionprops(labeled, intensity_image=intensity_image),
        key=lambda r: region_float_property(r, "intensity_max", "max_intensity"),
        reverse=True,
    )
    rows: list[dict[str, float | str]] = []
    limited_regions = regions[:max_candidates]
    for region in limited_regions:
        bg_mean, bg_std = local_background_stats_for_region(
            intensity_image,
            labeled,
            int(region.label),
            region.bbox,
        )
        min_row, min_col, max_row, max_col = region.bbox
        local_label = labeled[min_row:max_row, min_col:max_col] == region.label
        local_enhanced = enhanced[min_row:max_row, min_col:max_col]
        candidate_mean = region_float_property(region, "intensity_mean", "mean_intensity")
        candidate_max = region_float_property(region, "intensity_max", "max_intensity")
        sbr = candidate_mean / max(bg_mean, 1e-9)
        cnr = (candidate_mean - bg_mean) / max(bg_std, 1e-9)
        major_axis = region_float_property(region, "axis_major_length", "major_axis_length")
        minor_axis_value = region_float_property(region, "axis_minor_length", "minor_axis_length")
        minor_axis = max(minor_axis_value, 1e-9)
        rows.append(
            {
                "group": group,
                "machine": machine,
                "file": source_file,
                "label": int(region.label),
                "area": float(region.area),
                "eccentricity": float(region.eccentricity),
                "major_axis_length": major_axis,
                "minor_axis_length": minor_axis_value,
                "axis_ratio": major_axis / minor_axis,
                "solidity": float(region.solidity),
                "mean_intensity": candidate_mean,
                "max_intensity": candidate_max,
                "enhanced_mean": float(np.mean(local_enhanced[local_label])),
                "local_background_mean": bg_mean,
                "local_background_std": bg_std,
                "sbr": float(sbr),
                "cnr": float(cnr),
            }
        )
    limited_mask = np.isin(labeled, [region.label for region in limited_regions])
    return rows, limited_mask


def count_local_peaks(enhanced: np.ndarray, min_distance: int = 3, threshold_abs: float | None = None) -> int:
    """Count local maxima in an enhanced image as a weak signal proxy.

    Args:
        enhanced (np.ndarray): Enhanced response image.
        min_distance (int): Minimum peak separation in pixels.
        threshold_abs (float | None): Optional absolute peak threshold.

    Returns:
        int: Number of local maxima.
    """

    threshold = candidate_threshold(enhanced) if threshold_abs is None else threshold_abs
    peaks = peak_local_max(enhanced, min_distance=min_distance, threshold_abs=threshold)
    return int(len(peaks))


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float] = (1.0, 0.1, 0.1)) -> np.ndarray:
    """Create an RGB overlay of candidate mask boundaries on a grayscale image.

    Args:
        image (np.ndarray): Base grayscale image.
        mask (np.ndarray): Boolean candidate mask.
        color (tuple[float, float, float]): RGB overlay color.

    Returns:
        np.ndarray: RGB image with candidate boundaries highlighted.
    """

    base = gray2rgb(exposure.rescale_intensity(image))
    eroded = ndi.binary_erosion(mask)
    boundary = mask & ~eroded
    for channel, value in enumerate(color):
        base[..., channel] = np.where(boundary, value, base[..., channel])
    return base


def show_image(ax: Axes, image: np.ndarray, title: str, cmap: str | None = "gray") -> None:
    """Display an image with a compact title and no axes.

    Args:
        ax (Axes): Matplotlib axes to draw into.
        image (np.ndarray): Image to display.
        title (str): Axes title.
        cmap (str | None): Matplotlib colormap name, or None for RGB images.

    Returns:
        None: This function draws into the provided axes.
    """

    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save and close a matplotlib figure as PNG.

    Args:
        fig (plt.Figure): Figure object to save.
        path (Path): Destination file path. The suffix is normalized to .png.

    Returns:
        None: The figure is written to disk and closed.
    """

    path = path.with_suffix(FIGURE_EXT)
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, format="png", bbox_inches="tight")
    plt.close(fig)


def image_to_uint16(image: np.ndarray) -> np.ndarray:
    """Convert an image-like array to uint16 for TIFF storage.

    Args:
        image (np.ndarray): Input image or mask.

    Returns:
        np.ndarray: TIFF-friendly uint16 image array.
    """

    if image.dtype == bool:
        return image.astype(np.uint16) * np.uint16(65535)
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.uint16)
    clipped = np.clip(image, 0.0, 1.0)
    return np.round(clipped * 65535.0).astype(np.uint16)


def save_image_tiff(image: np.ndarray, path: Path) -> Path:
    """Save an image or mask array as a TIFF file.

    Args:
        image (np.ndarray): Image or mask to save.
        path (Path): Destination path. The suffix is normalized to .tif.

    Returns:
        Path: The TIFF path written to disk.
    """

    path = path.with_suffix(TIFF_IMAGE_EXT)
    ensure_dir(path.parent)
    imwrite(path, image_to_uint16(image))
    return path


def sanitized_stem(path: Path) -> str:
    """Convert a file path into a safe output stem.

    Args:
        path (Path): Source image path.

    Returns:
        str: Filename stem safe for output names.
    """

    return path.stem.replace(" ", "_").replace("/", "_")
