from pathlib import Path
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_DIR = Path("../modelsOpNav/img")
TARGET_TIME = 240.1

BACKGROUND_ALPHA = 0.35
THRESHOLD_SIGMA = 5.0
WARMUP_FRAMES = 10
MINIMUM_AREA = 2
MAXIMUM_AREA = 5000
BLUR_SIZE = 1
FRAME_INTERVAL_NS = 100_000_000
FREEZE_DURATION_NS = 198_000_000

# Find and sort images by their timestamp
paths = list(IMAGE_DIR.glob("cameraImages*.png"))

def image_time(path):
    match = re.search(r"cameraImages([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(match.group(1)) if match else None

paths = [(image_time(p), p) for p in paths]
paths = [(t, p) for t, p in paths if t is not None]
paths.sort(key=lambda item: item[0])

if not paths:
    raise FileNotFoundError(f"No cameraImages*.png files found in {IMAGE_DIR.resolve()}")

target_index = min(
    range(len(paths)),
    key=lambda i: abs(paths[i][0] - TARGET_TIME)
)

target_time, target_path = paths[target_index]

print(f"Found {len(paths)} images")
print(f"Selected: {target_path.name} ({target_time:.6f} s)")

background = None
warmup_counter = 0
candidate_active = False
freeze_until_ns = 0

target_original = None
target_background = None
target_difference = None
target_mask = None
target_candidates = []

for i, (time_s, path) in enumerate(paths[:target_index + 1]):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        print(f"Skipping unreadable image: {path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if BLUR_SIZE > 1:
        gray = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

    gray_float = gray.astype(np.float32)

    if background is None or background.shape != gray_float.shape:
        background = gray_float.copy()
        warmup_counter = 0

    if warmup_counter < WARMUP_FRAMES:
        background = gray_float.copy()
        warmup_counter += 1
        candidate_active = False
        continue

    # Save the exact background BEFORE processing this frame.
    background_before_detection = background.copy()

    # Positive difference
    difference = np.maximum(
        gray_float - background_before_detection, 0
    )

    # Column-based adaptive threshold
    column_means = np.mean(difference, axis=0)
    column_stds = np.std(difference, axis=0, ddof=1)

    image_mean_value = np.mean(column_means)
    maximum_column_std = np.max(column_stds)

    adaptive_threshold = (
    image_mean_value + THRESHOLD_SIGMA * maximum_column_std)
    threshold = max(adaptive_threshold, 25.0)

    mask = (difference > threshold).astype(np.uint8)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    candidates = []

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if MINIMUM_AREA <= area <= MAXIMUM_AREA:
            candidates.append({
                "label": label,
                "area": area,
                "x": float(centroids[label][0]),
                "y": float(centroids[label][1]),
                "radius": float(np.sqrt(area / np.pi))
            })

    candidates.sort(key=lambda c: c["area"], reverse=True)

    # Reproduce the C++ candidate-triggered freeze
    candidate_detected = len(candidates) > 0

    current_time_ns = i * FRAME_INTERVAL_NS

    if candidate_detected and not candidate_active:
        freeze_until_ns = current_time_ns + FREEZE_DURATION_NS

    candidate_active = candidate_detected

    # Update only AFTER detection, as in the C++ implementation
    if current_time_ns >= freeze_until_ns:
        cv2.accumulateWeighted(
            gray_float, background, BACKGROUND_ALPHA
        )

    if i == target_index:
        target_original = image.copy()
        target_background = background_before_detection.copy()
        target_difference = difference.copy()
        target_mask = mask.copy()
        target_labels = labels.copy()
        target_candidates = candidates
        target_threshold = threshold
        target_frame_index = i
        target_freeze_until = freeze_until_ns

if target_original is None:
    raise RuntimeError("Could not reconstruct the selected frame.")

# Keep only accepted components in the displayed mask
candidate_mask = np.zeros_like(target_mask, dtype=np.uint8)

for candidate in target_candidates:
    candidate_mask[target_labels == candidate["label"]] = 255

# Draw candidates on the original image
overlay = target_original.copy()

contours, _ = cv2.findContours(
    candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

for candidate in target_candidates:
    x = int(round(candidate["x"]))
    y = int(round(candidate["y"]))
    cv2.circle(overlay, (x, y), 3, (0, 255, 0), 1)

print(f"\nFrame index: {target_frame_index}")
print(f"Threshold: {target_threshold:.3f}")
print(f"Candidates: {len(target_candidates)}")
print(f"Freeze-until time: {target_freeze_until / 1e9:.3f} s")

for j, c in enumerate(target_candidates[:15], start=1):
    print(
        f"{j:2d}: area={c['area']:4d}, "
        f"x={c['x']:.1f}, y={c['y']:.1f}, "
        f"radius={c['radius']:.2f}"
    )

# Display all five diagnostic panels
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(cv2.cvtColor(target_original, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original image at 58 s")

axes[0, 1].imshow(target_background, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("Background used at 58 s")

axes[0, 2].imshow(target_difference, cmap="gray")
axes[0, 2].set_title("Positive difference")

axes[1, 0].imshow(target_mask, cmap="gray")
axes[1, 0].set_title("Threshold mask")

axes[1, 1].imshow(candidate_mask, cmap="gray")
axes[1, 1].set_title("Accepted candidate regions")

axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Candidates overlaid on original")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

output_path = Path("impact_flash_diagnostic.png")
plt.savefig(output_path, dpi=160, bbox_inches="tight")
print(f"\nSaved: {output_path.resolve()}")

plt.show()
