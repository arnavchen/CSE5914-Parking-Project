from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import convolve
from skimage import color, filters, io, measure, util


@dataclass(frozen=True)
class DetectorConfig:
    original_image_path: str = "2012-09-12_06_05_16_jpg.rf.aa059af912ba1103c6fd330f06b04e3d.jpg"
    rectified_image_path: str = "rectified_parking_lot.png"
    homography_metadata_path: str = "homography_metadata.json"

    vertical_kernel: tuple[tuple[float, ...], ...] = (
        (-1.0, 2.0, -1.0),
        (-1.0, 2.0, -1.0),
        (-1.0, 2.0, -1.0),
        (-1.0, 2.0, -1.0),
        (-1.0, 2.0, -1.0),
    )
    gaussian_sigma: float = 0.8
    gray_floor_percentile: float = 46.0
    response_percentile: float = 78.0
    min_line_response: float = 0.04
    cleanup_kernel_size: int = 3
    cleanup_min_neighbors: int = 3
    vertical_gap_threshold: int = 30
    horizontal_connect_tolerance: int = 2
    min_component_area: int = 20
    min_component_height: int = 8
    max_component_width: int = 12
    row_center_tolerance: int = 40
    min_vertical_overlap: int = 8
    width_ratio_min: float = 0.6
    width_ratio_max: float = 1.4
    height_ratio_min: float = 0.65
    height_ratio_max: float = 1.35

    overlay_color: tuple[int, int, int] = (255, 64, 64)
    spot_color: tuple[int, int, int] = (64, 220, 255)


CONFIG = DetectorConfig()


def cleanup_mask(mask: np.ndarray) -> np.ndarray:
    neighbor_count = convolve(
        mask.astype(np.float32),
        np.ones((CONFIG.cleanup_kernel_size, CONFIG.cleanup_kernel_size), dtype=np.float32),
        mode="reflect",
    )
    return (mask > 0) & (neighbor_count >= CONFIG.cleanup_min_neighbors)


def connect_vertical_gaps(mask: np.ndarray) -> np.ndarray:
    connected = mask.astype(bool).copy()
    height, width = connected.shape

    for y in range(1, height):
        xs = np.flatnonzero(connected[y])
        for x in xs:
            search_y1 = max(0, y - CONFIG.vertical_gap_threshold)
            found = False
            for prev_y in range(y - 1, search_y1 - 1, -1):
                x1 = max(0, x - CONFIG.horizontal_connect_tolerance)
                x2 = min(width, x + CONFIG.horizontal_connect_tolerance + 1)
                above_xs = np.flatnonzero(connected[prev_y, x1:x2])
                if above_xs.size == 0:
                    continue

                best_x = x1 + int(above_xs[np.argmin(np.abs((x1 + above_xs) - x))])
                fill_x1 = min(x, best_x)
                fill_x2 = max(x, best_x) + 1
                connected[prev_y:y + 1, fill_x1:fill_x2] = True
                found = True
                break
            if found:
                continue

    filtered = np.zeros_like(connected, dtype=bool)
    for region in measure.regionprops(measure.label(connected)):
        min_row, min_col, max_row, max_col = region.bbox
        height = max_row - min_row
        width = max_col - min_col
        if region.area < CONFIG.min_component_area:
            continue
        if height < CONFIG.min_component_height:
            continue
        if width > CONFIG.max_component_width:
            continue
        filtered[region.slice] |= region.image

    return filtered.astype(np.uint8)


def extract_line_segments(mask: np.ndarray) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for region in measure.regionprops(measure.label(mask.astype(bool))):
        min_row, min_col, max_row, max_col = region.bbox
        segments.append(
            {
                "id": int(region.label),
                "bbox": [int(min_col), int(min_row), int(max_col), int(max_row)],
                "x_center": float(region.centroid[1]),
                "y_center": float(region.centroid[0]),
                "height": int(max_row - min_row),
                "width": int(max_col - min_col),
                "area": int(region.area),
            }
        )
    return segments


def group_lines_by_row(segments: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for segment in sorted(segments, key=lambda item: item["y_center"]):
        placed = False
        for group in groups:
            group_center = float(np.mean([item["y_center"] for item in group]))
            if abs(segment["y_center"] - group_center) <= CONFIG.row_center_tolerance:
                group.append(segment)
                placed = True
                break
        if not placed:
            groups.append([segment])

    for group in groups:
        group.sort(key=lambda item: item["x_center"])
    return groups


def build_parking_spots(line_groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    spots: list[dict[str, Any]] = []
    spot_id = 0

    for group_id, group in enumerate(line_groups):
        candidates: list[dict[str, Any]] = []

        for left, right in zip(group, group[1:]):
            left_x1, left_y1, left_x2, left_y2 = left["bbox"]
            right_x1, right_y1, right_x2, right_y2 = right["bbox"]

            overlap_y1 = max(left_y1, right_y1)
            overlap_y2 = min(left_y2, right_y2)
            overlap_height = overlap_y2 - overlap_y1
            spot_width = right_x1 - left_x2

            if overlap_height < CONFIG.min_vertical_overlap:
                continue
            if spot_width <= 0:
                continue

            candidates.append(
                {
                    "group_id": group_id,
                    "left_line_id": left["id"],
                    "right_line_id": right["id"],
                    "bbox": [
                        int(left_x2),
                        int(overlap_y1),
                        int(right_x1),
                        int(overlap_y2),
                    ],
                    "width": int(spot_width),
                    "height": int(overlap_height),
                    "left_height": int(left["height"]),
                    "right_height": int(right["height"]),
                }
            )

        if not candidates:
            continue

        typical_width = float(np.median([item["width"] for item in candidates]))
        typical_height = float(np.median([item["height"] for item in candidates]))

        for candidate in candidates:
            if not (CONFIG.width_ratio_min * typical_width <= candidate["width"] <= CONFIG.width_ratio_max * typical_width):
                continue
            if not (CONFIG.height_ratio_min * typical_height <= candidate["height"] <= CONFIG.height_ratio_max * typical_height):
                continue
            if not (CONFIG.height_ratio_min * typical_height <= candidate["left_height"] <= CONFIG.height_ratio_max * typical_height):
                continue
            if not (CONFIG.height_ratio_min * typical_height <= candidate["right_height"] <= CONFIG.height_ratio_max * typical_height):
                continue

            candidate["id"] = spot_id
            spots.append(candidate)
            spot_id += 1

    return spots


def detect_white_lines(image: np.ndarray) -> np.ndarray:
    gray = color.rgb2gray(image)
    gray = filters.gaussian(gray, sigma=CONFIG.gaussian_sigma, preserve_range=True)
    vertical_response = np.clip(
        convolve(
            gray,
            np.asarray(CONFIG.vertical_kernel, dtype=np.float32),
            mode="reflect",
        ),
        0.0,
        None,
    )
    line_response = vertical_response

    gray_floor = np.percentile(gray, CONFIG.gray_floor_percentile)
    response_floor = max(
        CONFIG.min_line_response,
        float(np.percentile(line_response, CONFIG.response_percentile)),
    )

    mask = (
        (gray >= gray_floor)
        & (line_response >= response_floor)
    )
    mask = cleanup_mask(mask)
    mask = connect_vertical_gaps(mask)

    return mask.astype(np.uint8)


def save_mask(mask: np.ndarray, output_path: str | Path) -> None:
    io.imsave(str(output_path), (mask * 255).astype(np.uint8), check_contrast=False)


def save_overlay(image: np.ndarray, mask: np.ndarray, output_path: str | Path) -> None:
    overlay = image.copy()
    overlay[mask.astype(bool)] = CONFIG.overlay_color
    io.imsave(str(output_path), overlay.astype(np.uint8), check_contrast=False)


def save_spot_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    spots: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    overlay = image.copy()
    overlay[mask.astype(bool)] = CONFIG.overlay_color

    for spot in spots:
        x1, y1, x2, y2 = spot["bbox"]
        overlay[y1:y2, max(0, x1 - 1):min(overlay.shape[1], x1 + 1)] = CONFIG.spot_color
        overlay[y1:y2, max(0, x2 - 1):min(overlay.shape[1], x2 + 1)] = CONFIG.spot_color
        overlay[max(0, y1 - 1):min(overlay.shape[0], y1 + 1), x1:x2] = CONFIG.spot_color
        overlay[max(0, y2 - 1):min(overlay.shape[0], y2 + 1), x1:x2] = CONFIG.spot_color

    io.imsave(str(output_path), overlay.astype(np.uint8), check_contrast=False)


def save_summary(
    image_path: Path,
    mask: np.ndarray,
    line_segments: list[dict[str, Any]],
    parking_spots: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    labeled = measure.label(mask.astype(bool))
    payload = {
        "image": str(image_path),
        "image_height": int(mask.shape[0]),
        "image_width": int(mask.shape[1]),
        "white_line_pixel_count": int(mask.sum()),
        "white_line_pixel_ratio": float(mask.mean()),
        "connected_line_count": int(labeled.max()),
        "line_segments": line_segments,
        "parking_spots": parking_spots,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2))


def main() -> None:
    rectified_path = Path(CONFIG.rectified_image_path)
    metadata_path = Path(CONFIG.homography_metadata_path)
    image_path = (
        rectified_path
        if rectified_path.exists() and metadata_path.exists()
        else Path(CONFIG.original_image_path)
    )

    image = io.imread(str(image_path))
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = util.img_as_ubyte(image).astype(np.uint8)

    mask = detect_white_lines(image)
    line_segments = extract_line_segments(mask)
    line_groups = group_lines_by_row(line_segments)
    parking_spots = build_parking_spots(line_groups)

    save_mask(mask, "white_line_mask.png")
    save_overlay(image, mask, "white_line_overlay.png")
    save_spot_overlay(image, mask, parking_spots, "parking_spot_overlay.png")
    save_summary(image_path, mask, line_segments, parking_spots, "white_line_summary.json")

    print(f"Input image: {image_path}")
    print(f"Detected white-line pixels: {int(mask.sum())}")
    print(f"Connected line segments: {len(line_segments)}")
    print(f"Parking spots from line gaps: {len(parking_spots)}")
    print("Saved mask to white_line_mask.png")
    print("Saved overlay to white_line_overlay.png")
    print("Saved parking spot overlay to parking_spot_overlay.png")
    print("Saved summary to white_line_summary.json")


if __name__ == "__main__":
    main()
