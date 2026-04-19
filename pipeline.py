from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from skimage import io, util

from parking_detection import detect_occupancy, visualize_results
from parking_homography import warp_image


TEST_IMAGE_PATH = "test.jpg"
HOMOGRAPHY_METADATA_PATH = "homography_metadata.json"
ROI_SUMMARY_PATH = "white_line_summary.json"

RECTIFIED_EMPTY_PATH = "rectified_empty_reference.png"
RECTIFIED_TEST_PATH = "rectified_test.png"
OCCUPANCY_OVERLAY_PATH = "parking_occupancy_overlay.png"
RESULTS_PATH = "parking_occupancy_results.json"


def main() -> None:
    metadata = json.loads(Path(HOMOGRAPHY_METADATA_PATH).read_text())
    roi_summary = json.loads(Path(ROI_SUMMARY_PATH).read_text())

    homography = np.asarray(metadata["homography"], dtype=np.float64)
    output_width = int(round(metadata["target_points"][1][0] + 1))
    output_height = int(round(metadata["target_points"][2][1] + 1))

    empty_image = io.imread(metadata["image_path"])
    test_image = io.imread(TEST_IMAGE_PATH)

    if empty_image.ndim == 2:
        empty_image = np.stack([empty_image, empty_image, empty_image], axis=-1)
    if test_image.ndim == 2:
        test_image = np.stack([test_image, test_image, test_image], axis=-1)
    if empty_image.shape[-1] == 4:
        empty_image = empty_image[..., :3]
    if test_image.shape[-1] == 4:
        test_image = test_image[..., :3]

    empty_image = util.img_as_ubyte(empty_image).astype(np.uint8)
    test_image = util.img_as_ubyte(test_image).astype(np.uint8)

    rectified_empty = warp_image(empty_image, homography, output_width, output_height)
    rectified_test = warp_image(test_image, homography, output_width, output_height)

    io.imsave(RECTIFIED_EMPTY_PATH, rectified_empty, check_contrast=False)
    io.imsave(RECTIFIED_TEST_PATH, rectified_test, check_contrast=False)

    rois = [tuple(spot["bbox"]) for spot in roi_summary["parking_spots"]]
    results = detect_occupancy(RECTIFIED_TEST_PATH, RECTIFIED_EMPTY_PATH, rois)
    visualize_results(RECTIFIED_TEST_PATH, results, OCCUPANCY_OVERLAY_PATH)

    spots = [
        {
            "id": spot["id"],
            "group_id": spot["group_id"],
            "bbox": list(result["bbox"]),
            "changed_ratio": round(float(result["changed_ratio"]), 4),
            "occupied": bool(result["is_occupied"]),
            "status": str(result["status"]).lower(),
        }
        for spot, result in zip(roi_summary["parking_spots"], results)
    ]

    Path(RESULTS_PATH).write_text(
        json.dumps(
            {
                "test_image": TEST_IMAGE_PATH,
                "rectified_test_image": RECTIFIED_TEST_PATH,
                "rectified_empty_image": RECTIFIED_EMPTY_PATH,
                "spots": spots,
            },
            indent=2,
        )
    )

    occupied_count = sum(1 for item in spots if item["occupied"])
    print(f"Rectified empty image saved to {RECTIFIED_EMPTY_PATH}")
    print(f"Rectified test image saved to {RECTIFIED_TEST_PATH}")
    print(f"Occupancy overlay saved to {OCCUPANCY_OVERLAY_PATH}")
    print(f"Occupancy results saved to {RESULTS_PATH}")
    print(f"Occupied spots: {occupied_count}/{len(spots)}")


if __name__ == "__main__":
    main()
