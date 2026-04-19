from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Config:
    BLUR_SIGMA = 1.0
    DIFF_THRESHOLD = 30
    OCCUPANCY_THRESHOLD = 0.15

    MORPH_KERNEL_SIZE = 3
    MORPH_ITERATIONS = 1

    EMPTY_COLOR = (0, 255, 0)
    OCCUPIED_COLOR = (255, 0, 0)
    BOX_THICKNESS = 2


def load_rgb_uint8(image_path: str) -> np.ndarray:
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = mpimg.imread(image_path)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.shape[-1] == 4:
        img = img[..., :3]

    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    return img


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)

    r = image[..., 0].astype(np.float32)
    g = image[..., 1].astype(np.float32)
    b = image[..., 2].astype(np.float32)

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def convolve1d_reflect(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(image, pad_width, mode="reflect")
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), axis, padded)


def gaussian_blur_gray(image: np.ndarray, sigma: float) -> np.ndarray:
    image = image.astype(np.float32)
    kernel = gaussian_kernel1d(sigma)
    out = convolve1d_reflect(image, kernel, axis=0)
    out = convolve1d_reflect(out, kernel, axis=1)
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess_image(image_path: str) -> np.ndarray:
    img = load_rgb_uint8(image_path)
    gray = to_gray(img)
    gray = gaussian_blur_gray(gray, Config.BLUR_SIGMA)
    return gray


def compute_foreground_mask(current_gray: np.ndarray, reference_gray: np.ndarray) -> np.ndarray:
    diff = np.abs(current_gray.astype(np.int16) - reference_gray.astype(np.int16)).astype(np.uint8)
    mask = (diff > Config.DIFF_THRESHOLD).astype(np.uint8)
    return mask


def binary_erosion(binary: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(binary, ((pad, pad), (pad, pad)), mode="constant")
    out = np.zeros_like(binary)

    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            window = padded[y:y + kernel_size, x:x + kernel_size]
            out[y, x] = 1 if np.all(window == 1) else 0

    return out


def binary_dilation(binary: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(binary, ((pad, pad), (pad, pad)), mode="constant")
    out = np.zeros_like(binary)

    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            window = padded[y:y + kernel_size, x:x + kernel_size]
            out[y, x] = 1 if np.any(window == 1) else 0

    return out


def apply_morphology(mask: np.ndarray) -> np.ndarray:
    binary = mask.copy()

    for _ in range(Config.MORPH_ITERATIONS):
        binary = binary_erosion(binary, Config.MORPH_KERNEL_SIZE)
        binary = binary_dilation(binary, Config.MORPH_KERNEL_SIZE)

    for _ in range(Config.MORPH_ITERATIONS):
        binary = binary_dilation(binary, Config.MORPH_KERNEL_SIZE)
        binary = binary_erosion(binary, Config.MORPH_KERNEL_SIZE)

    return binary


def classify_parking_spots(mask: np.ndarray, rois: list[tuple[int, int, int, int]]) -> list[dict]:
    results = []

    for i, (x1, y1, x2, y2) in enumerate(rois):
        roi_mask = mask[y1:y2, x1:x2]

        if roi_mask.size == 0:
            changed_ratio = 0.0
        else:
            changed_ratio = np.count_nonzero(roi_mask) / roi_mask.size

        is_occupied = changed_ratio > Config.OCCUPANCY_THRESHOLD

        results.append({
            "roi_id": i,
            "bbox": (x1, y1, x2, y2),
            "changed_ratio": changed_ratio,
            "is_occupied": is_occupied,
            "status": "OCCUPIED" if is_occupied else "EMPTY",
        })

    return results


def draw_box(image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
             color: tuple[int, int, int], thickness: int) -> None:
    h, w = image.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    for t in range(thickness):
        if y1 + t < h:
            image[y1 + t, x1:x2 + 1] = color
        if y2 - t >= 0:
            image[y2 - t, x1:x2 + 1] = color
        if x1 + t < w:
            image[y1:y2 + 1, x1 + t] = color
        if x2 - t >= 0:
            image[y1:y2 + 1, x2 - t] = color


def visualize_results(image_path: str, results: list[dict], output_path: str = "result.png") -> None:
    img = load_rgb_uint8(image_path)
    vis = img.copy()

    for result in results:
        x1, y1, x2, y2 = result["bbox"]
        color = Config.OCCUPIED_COLOR if result["is_occupied"] else Config.EMPTY_COLOR
        draw_box(vis, x1, y1, x2, y2, color, Config.BOX_THICKNESS)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(vis)
    ax.axis("off")

    for result in results:
        x1, y1, _, _ = result["bbox"]
        label = f'{result["status"]} ({result["changed_ratio"]:.2f})'
        ax.text(
            x1,
            max(0, y1 - 5),
            label,
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none")
        )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def detect_occupancy(current_image_path: str,
                     reference_image_path: str,
                     rois: list[tuple[int, int, int, int]]) -> list[dict]:
    current_gray = preprocess_image(current_image_path)
    reference_gray = preprocess_image(reference_image_path)

    if current_gray.shape != reference_gray.shape:
        raise ValueError("Current image and reference image must have the same size.")

    mask = compute_foreground_mask(current_gray, reference_gray)
    mask = apply_morphology(mask)
    results = classify_parking_spots(mask, rois)
    return results


if __name__ == "__main__":
    rois = [
        (50, 50, 150, 130),
        (170, 50, 270, 130),
        (290, 50, 390, 130),
        (50, 170, 150, 250),
        (170, 170, 270, 250),
    ]

    current_image = "current_parking_lot.jpg"
    reference_image = "empty_lot_reference.jpg"

    results = detect_occupancy(current_image, reference_image, rois)

    for result in results:
        print(f'ROI {result["roi_id"]}: {result["status"]} ({result["changed_ratio"]:.2%})')

    visualize_results(current_image, results, "occupancy_result.png")
    print("Saved visualization to occupancy_result.png")