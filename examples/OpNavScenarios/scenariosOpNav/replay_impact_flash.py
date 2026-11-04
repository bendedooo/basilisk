
from pathlib import Path
import re
import csv
import cv2
import numpy as np
import math

# ------------------------------------------------------------
# Configuration: must match ImpactFlashDetection.cpp
# ------------------------------------------------------------

IMAGE_DIR = Path("../modelsOpNav/img")
OUTPUT_FILE = IMAGE_DIR / "impact_flash_replay.csv"

BACKGROUND_ALPHA = 0.35
THRESHOLD_SIGMA = 5.0
HIGH_THRESHOLD_RATIO = 1.0
LOW_THRESHOLD_RATIO = 1.0

WARMUP_FRAMES = 30          # frames used to learn background AND noise

# Adaptive per-pixel threshold (replaces mu + 5 sigma and the fixed 25 floor)
NOISE_BETA = 0.05           # memory of the per-pixel noise variance (~1/beta frames)
K_HIGH = 6.0                # seed threshold, in units of local noise sigma
K_LOW = 3.0                 # region-growing threshold (hysteresis), in sigma
SIGMA_MIN = 0.5             # gray levels; uint8 quantization floor
VARIANCE_WINDOW = 5         # spatial smoothing of the variance map (px)
SIGMA_DILATE = 1            # local max of sigma over (2n+1)^2 px (moving edges)
LIMB_MARGIN = 4             # reject candidates whose peak is this close to the limb (thesis 3.2.2)
DISC_FRACTION = 0.05        # disc = background > fraction * 99.5th percentile

MINIMUM_AREA = 2
MAXIMUM_AREA = 5000
BLUR_SIZE = 1
NOISE_SF = 2.0

# Match the Basilisk message definition.
# Set this to the MAX_CIRCLE_NUM value in your C++ header.
MAX_CIRCLE_NUM = 10

# Match the C++ freeze duration.
BACKGROUND_FREEZE_NS = 198_000_000

# Diagnostic figures (original, background, difference, threshold mask,
# accepted candidates, candidates on original)
SAVE_FIGURES = True
FIGURE_TIMES = [58.0]          # extra times (s) to plot even without detections
FIGURE_DIR = Path("flash_figures")


# ------------------------------------------------------------
# Image ordering
# ------------------------------------------------------------

def image_time(path):
    match = re.search(r"(\d+(?:\.\d+)?)$", path.stem)

    if match is None:
        raise ValueError(f"Cannot extract timestamp from {path.name}")

    return float(match.group(1))


image_paths = sorted(
    [
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    ],
    key=image_time
)

if not image_paths:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")


# ------------------------------------------------------------
# Diagnostic figure
# ------------------------------------------------------------

figure_frame_names = set()
if FIGURE_TIMES:
    for wanted in FIGURE_TIMES:
        figure_frame_names.add(
            min(image_paths, key=lambda p: abs(image_time(p) - wanted)).name
        )


def save_diagnostic_figure(frame, background, positive_difference,
                           local_sigma, snr, labels, candidates, time_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(exist_ok=True)

    typical_threshold = K_HIGH * float(np.median(local_sigma))
    disc = background > DISC_FRACTION * float(np.percentile(background, 99.5))
    if disc.any():
        typical_threshold = K_HIGH * float(np.median(local_sigma[disc]))
    difference_vmax = max(10.0, 2.0 * typical_threshold)

    # 0 = below, 0.5 = above K_LOW sigma, 1 = above K_HIGH sigma
    threshold_mask = 0.5 * (snr > K_LOW) + 0.5 * (snr > K_HIGH)

    accepted_mask = np.zeros(labels.shape, np.uint8)
    for candidate in candidates:
        accepted_mask[labels == candidate["label"]] = 255

    overlay = frame.copy()
    contours, _ = cv2.findContours(
        accepted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
    for index, candidate in enumerate(candidates):
        x, y = int(round(candidate["cx"])), int(round(candidate["cy"]))
        cv2.circle(overlay, (x, y), 10, (0, 255, 0), 1)
        cv2.putText(overlay, f"#{index}", (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Original image at {time_s:.2f} s")

    axes[0, 1].imshow(background, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Background used at {time_s:.2f} s")

    im = axes[0, 2].imshow(positive_difference, cmap="inferno",
                           vmin=0, vmax=difference_vmax)
    axes[0, 2].set_title(f"Positive difference (0-{difference_vmax:.0f} gray levels)")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(threshold_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Threshold mask (gray > {K_LOW:g}σ, white > {K_HIGH:g}σ)\n"
                         f"typical threshold on disc ≈ {typical_threshold:.1f} gray levels")

    axes[1, 1].imshow(accepted_mask, cmap="gray")
    axes[1, 1].set_title(f"Accepted candidate regions ({len(candidates)})")

    axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Candidates overlaid on original")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    output = FIGURE_DIR / f"flash_{time_s:09.3f}s.png"
    plt.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {output}")
    for index, candidate in enumerate(candidates):
        peak = float(positive_difference[labels == candidate["label"]].max())
        peak_snr = float(snr[labels == candidate["label"]].max())
        print(f"  [{index}] x={candidate['cx']:.1f} y={candidate['cy']:.1f} "
              f"area={candidate['area']} peak_diff={peak:.1f} peak_snr={peak_snr:.1f}")


# ------------------------------------------------------------
# Detector state
# ------------------------------------------------------------

background_gray = None
warmup_counter = 0

background_freeze_until_ns = 0
candidate_active = False

noise_variance = None
results = []


# ------------------------------------------------------------
# Process frames
# ------------------------------------------------------------

for frame_index, image_path in enumerate(image_paths):

    # --------------------------------------------------------
    # 1. Decode image (equivalent to imdecode IMREAD_COLOR)
    # --------------------------------------------------------

    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if frame is None:
        print(f"Could not decode: {image_path}")
        continue

    # --------------------------------------------------------
    # 2. Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if BLUR_SIZE > 1:
        kernel = (
            BLUR_SIZE + 1
            if BLUR_SIZE % 2 == 0
            else BLUR_SIZE
        )

        gray = cv2.GaussianBlur(
            gray,
            (kernel, kernel),
            0.0
        )

    # --------------------------------------------------------
    # 3. Convert to float32
    # --------------------------------------------------------

    gray_float = gray.astype(np.float32)

    # --------------------------------------------------------
    # 4. Initialize background if necessary
    # --------------------------------------------------------

    if (
        background_gray is None
        or background_gray.shape != gray_float.shape
    ):
        background_gray = gray_float.copy()
        warmup_counter = 0

    # --------------------------------------------------------
    # 5. Warm-up: learn the background AND the per-pixel noise
    # --------------------------------------------------------

    if warmup_counter < WARMUP_FRAMES:

        if (
            warmup_counter == 0
            or noise_variance is None
            or noise_variance.shape != gray_float.shape
        ):
            background_gray = gray_float.copy()
            noise_variance = np.zeros_like(gray_float)
        else:
            warmup_difference = gray_float - background_gray
            noise_variance += (
                warmup_difference ** 2 - noise_variance
            ) / warmup_counter
            cv2.accumulateWeighted(
                gray_float,
                background_gray,
                BACKGROUND_ALPHA
            )

        warmup_counter += 1

        results.append({
            "frame": frame_index,
            "time_s": image_time(image_path),
            "image": image_path.name,
            "threshold": 0.0,
            "candidate_count": 0,
            "valid": 0,
            "detections": []
        })

        continue

    # --------------------------------------------------------
    # 6. Signed and positive difference image
    # --------------------------------------------------------

    difference_float = gray_float - background_gray

    positive_difference = np.maximum(
        difference_float,
        0.0
    )

    # --------------------------------------------------------
    # 7. Per-pixel adaptive threshold from the learned noise
    # --------------------------------------------------------

    smoothed_variance = cv2.blur(
        noise_variance,
        (VARIANCE_WINDOW, VARIANCE_WINDOW)
    )

    local_sigma = np.sqrt(
        np.maximum(smoothed_variance, SIGMA_MIN ** 2)
    )

    if SIGMA_DILATE > 0:
        dilate_kernel = np.ones(
            (2 * SIGMA_DILATE + 1, 2 * SIGMA_DILATE + 1), np.uint8
        )
        local_sigma = cv2.dilate(local_sigma, dilate_kernel)

    snr = difference_float / local_sigma

    # Moon disc from the background, shrunk by LIMB_MARGIN (thesis 3.2.2)
    disc_level = DISC_FRACTION * float(
        np.percentile(background_gray, 99.5)
    )
    disc_mask = (background_gray > disc_level).astype(np.uint8)
    erode_kernel = np.ones(
        (2 * LIMB_MARGIN + 1, 2 * LIMB_MARGIN + 1), np.uint8
    )
    disc_inside = cv2.erode(disc_mask, erode_kernel).astype(bool)

    # Reported threshold: median per-pixel seed threshold on the disc
    adaptive_threshold = (
        float(K_HIGH * np.median(local_sigma[disc_mask > 0]))
        if disc_mask.any()
        else 0.0
    )

    # --------------------------------------------------------
    # 8. High and low threshold masks (in units of local sigma)
    # --------------------------------------------------------

    high_mask = (snr > K_HIGH).astype(np.uint8)

    low_mask = (snr > K_LOW).astype(np.uint8)

    # --------------------------------------------------------
    # 9. Connected components on low threshold mask
    # --------------------------------------------------------

    component_count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            low_mask,
            connectivity=8,
            ltype=cv2.CV_32S
        )
    )

    candidates = []

    for label in range(1, component_count):

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if (
            area < MINIMUM_AREA
            or area > MAXIMUM_AREA
        ):
            continue

        # Match C++ componentMask = (labels == label)
        component_mask = (
            labels == label
        ).astype(np.uint8)

        # Match bitwise_and(componentMask, highMask)
        high_pixels = cv2.bitwise_and(
            component_mask,
            high_mask
        )

        if cv2.countNonZero(high_pixels) == 0:
            continue

        # Thesis 3.2.2: the strongest pixel must lie inside the disc
        peak_y, peak_x = np.unravel_index(
            np.argmax(np.where(component_mask > 0, snr, -np.inf)),
            snr.shape
        )

        if not disc_inside[peak_y, peak_x]:
            continue

        cx = float(centroids[label, 0])
        cy = float(centroids[label, 1])

        candidates.append({
            "label": label,
            "area": area,
            "cx": cx,
            "cy": cy
        })

    # --------------------------------------------------------
    # 10. Sort candidates by descending area
    # --------------------------------------------------------

    candidates.sort(
        key=lambda candidate: candidate["area"],
        reverse=True
    )

    # --------------------------------------------------------
    # 11. Limit output to MAX_CIRCLE_NUM
    # --------------------------------------------------------

    candidates = candidates[:MAX_CIRCLE_NUM]

    detections = []

    for candidate in candidates:

        equivalent_radius = math.sqrt(
            candidate["area"] / math.pi
        )

        detections.append({
            "planet_id": 2,
            "cx": candidate["cx"],
            "cy": candidate["cy"],
            "radius": equivalent_radius,
            "area": candidate["area"],
            "uncertainty": NOISE_SF
        })

    valid = int(len(detections) > 0)

    # --------------------------------------------------------
    # Diagnostic figure (before the background is updated)
    # --------------------------------------------------------

    if SAVE_FIGURES and (candidates or image_path.name in figure_frame_names):
        save_diagnostic_figure(
            frame, background_gray, positive_difference,
            local_sigma, snr, labels, candidates, image_time(image_path)
        )

    # --------------------------------------------------------
    # 12. Background freeze
    # --------------------------------------------------------

    # Use frame timestamps in nanoseconds.
    # If the image filenames contain no timestamps, this assumes
    # consecutive frames are separated by 0.1 seconds.
    current_time_s = image_time(image_path)
    current_sim_nanos = int(current_time_s * 1e9)

    candidate_detected = len(candidates) > 0

    if candidate_detected and not candidate_active:

        background_freeze_until_ns = (
            current_sim_nanos + BACKGROUND_FREEZE_NS
        )

    candidate_active = candidate_detected

    if current_sim_nanos >= background_freeze_until_ns:

        cv2.accumulateWeighted(
            gray_float,
            background_gray,
            BACKGROUND_ALPHA
        )
        noise_variance = (
            (1.0 - NOISE_BETA) * noise_variance
            + NOISE_BETA * difference_float ** 2
        ).astype(np.float32)

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "frame": frame_index,
        "time_s": image_time(image_path),
        "image": image_path.name,
        "threshold": adaptive_threshold,
        "candidate_count": len(candidates),
        "valid": valid,
        "detections": detections
    })


# ------------------------------------------------------------
# Write results
# One row per candidate; zero-detection frames are retained.
# ------------------------------------------------------------

with open(OUTPUT_FILE, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "frame",
        "time_s",
        "image",
        "threshold",
        "candidate_count",
        "valid",
        "candidate_index",
        "planet_id",
        "x",
        "y",
        "radius",
        "area",
        "uncertainty"
    ])

    for result in results:

        if not result["detections"]:

            writer.writerow([
                result["frame"],
                result["time_s"],
                result["image"],
                result["threshold"],
                result["candidate_count"],
                result["valid"],
                "",
                "",
                "",
                "",
                "",
                "",
                ""
            ])

        else:

            for candidate_index, detection in enumerate(
                result["detections"]
            ):

                writer.writerow([
                    result["frame"],
                    result["time_s"],
                    result["image"],
                    result["threshold"],
                    result["candidate_count"],
                    result["valid"],
                    candidate_index,
                    detection["planet_id"],
                    detection["cx"],
                    detection["cy"],
                    detection["radius"],
                    detection["area"],
                    detection["uncertainty"]
                ])


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

processed_frames = len(results)
valid_frames = sum(r["valid"] for r in results)

total_candidates = sum(
    r["candidate_count"] for r in results
)

print(f"Processed frames: {processed_frames}")
print(f"Valid frames: {valid_frames}")
print(f"Frames with no detections: {processed_frames - valid_frames}")
print(f"Total reported candidates: {total_candidates}")
print(f"Output: {OUTPUT_FILE}")

detection_times = [r["time_s"] for r in results if r["valid"]]
print(f"Detection times (s): {detection_times[:50]}"
      + (" ..." if len(detection_times) > 50 else ""))
