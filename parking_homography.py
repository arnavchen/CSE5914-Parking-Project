from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage import io, util


@dataclass(frozen=True)
class HomographyConfig:
    points_path: str = "homography_points.json"
    output_warp_path: str = "rectified_parking_lot.png"
    output_overlay_path: str = "homography_source_overlay.png"
    output_metadata_path: str = "homography_metadata.json"
    output_width: int = 900
    output_height: int = 700


CONFIG = HomographyConfig()
def load_source_points(points_path: str | Path) -> tuple[Path, np.ndarray]:
    path = Path(points_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Point file not found: {path}. Run pick_homography_points.py first."
        )

    payload = json.loads(path.read_text())
    image_path = Path(payload["image_path"])
    source_points = np.asarray(payload["source_points"], dtype=np.float64)
    if source_points.shape != (4, 2):
        raise ValueError("source_points must contain exactly 4 [x, y] pairs.")

    return image_path, source_points


def compute_homography(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    if source_points.shape != target_points.shape or source_points.shape[0] < 4:
        raise ValueError("Need at least four matching 2D points to compute a homography.")

    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source_points, target_points):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, x * u, y * u, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, x * v, y * v, v])

    a_matrix = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(a_matrix)
    homography = vh[-1].reshape(3, 3)

    if homography[2, 2] == 0:
        raise ValueError("Computed homography is singular.")

    return homography / homography[2, 2]


def apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    warped = homogeneous @ homography.T
    warped /= warped[:, 2:3]
    return warped[:, :2]


def bilinear_sample(image: np.ndarray, x: float, y: float) -> np.ndarray:
    height, width = image.shape[:2]
    if x < 0.0 or x > width - 1.0 or y < 0.0 or y > height - 1.0:
        return np.zeros(3, dtype=np.float64)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    dx = x - x0
    dy = y - y0

    top = (1.0 - dx) * image[y0, x0] + dx * image[y0, x1]
    bottom = (1.0 - dx) * image[y1, x0] + dx * image[y1, x1]
    return (1.0 - dy) * top + dy * bottom


def warp_image(image: np.ndarray, homography: np.ndarray, output_width: int, output_height: int) -> np.ndarray:
    inverse_homography = np.linalg.inv(homography)
    warped = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    for y in range(output_height):
        target_points = np.column_stack(
            [
                np.arange(output_width, dtype=np.float64),
                np.full(output_width, y, dtype=np.float64),
            ]
        )
        source_points = apply_homography(target_points, inverse_homography)

        row = np.zeros((output_width, 3), dtype=np.float64)
        for x, (src_x, src_y) in enumerate(source_points):
            row[x] = bilinear_sample(image, float(src_x), float(src_y))

        warped[y] = np.clip(row, 0, 255).astype(np.uint8)

    return warped


def save_source_overlay(image: np.ndarray, source_points: np.ndarray, output_path: str | Path) -> None:
    overlay = Image.fromarray(image)
    draw = ImageDraw.Draw(overlay)

    polygon = [tuple(point) for point in source_points.tolist()]
    draw.line(polygon + [polygon[0]], fill=(255, 80, 80), width=3)

    radius = 5
    for index, (x, y) in enumerate(polygon):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(64, 220, 255))
        draw.text((x + 8, y + 8), str(index), fill=(255, 255, 255))

    overlay.save(output_path)


def main() -> None:
    image_path, source_points = load_source_points(CONFIG.points_path)
    image = io.imread(str(image_path))
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = util.img_as_ubyte(image).astype(np.uint8)
    target_points = np.array(
        [
            [0.0, 0.0],
            [CONFIG.output_width - 1.0, 0.0],
            [CONFIG.output_width - 1.0, CONFIG.output_height - 1.0],
            [0.0, CONFIG.output_height - 1.0],
        ],
        dtype=np.float64,
    )

    homography = compute_homography(source_points, target_points)
    warped = warp_image(image, homography, CONFIG.output_width, CONFIG.output_height)

    io.imsave(str(CONFIG.output_warp_path), warped, check_contrast=False)
    save_source_overlay(image, source_points, CONFIG.output_overlay_path)

    metadata = {
        "image_path": str(image_path),
        "source_points": source_points.tolist(),
        "target_points": target_points.tolist(),
        "homography": homography.tolist(),
    }
    Path(CONFIG.output_metadata_path).write_text(json.dumps(metadata, indent=2))

    print(f"Saved rectified image to {CONFIG.output_warp_path}")
    print(f"Saved source overlay to {CONFIG.output_overlay_path}")
    print(f"Saved homography metadata to {CONFIG.output_metadata_path}")


if __name__ == "__main__":
    main()
