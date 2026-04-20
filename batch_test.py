from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

TEST_IMAGES_DIR = Path("parking_lot_test_images")
RESULTS_DIR = Path("batch_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Find all test images with 2012 in the name
test_images = sorted(TEST_IMAGES_DIR.glob("2012*.jpg"))

results_summary = []

for image_path in test_images:
    test_file = Path("test.jpg")
    
    print(f"\nTesting: {image_path.name}")
    
    # Copy image to test.jpg
    shutil.copy(image_path, test_file)
    
    # Run pipeline using the current venv's Python
    result = subprocess.run([sys.executable, "pipeline.py"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr[:100]}")
        results_summary.append({
            "image": image_path.name,
            "status": "FAILED",
            "error": result.stderr[:200]
        })
        continue
    
    # Read results
    results_path = Path("parking_occupancy_results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        
        occupied_count = sum(1 for spot in results["spots"] if spot["occupied"])
        empty_count = len(results["spots"]) - occupied_count
        
        # Copy results overlay to batch_results
        overlay_src = Path("parking_occupancy_overlay.png")
        if overlay_src.exists():
            output_name = image_path.stem + "_results.png"
            shutil.copy(overlay_src, RESULTS_DIR / output_name)
        
        print(f"✓ Occupied: {occupied_count:2d}/{len(results['spots']):2d} | Empty: {empty_count:2d}/{len(results['spots']):2d}")
        
        results_summary.append({
            "image": image_path.name,
            "status": "SUCCESS",
            "total_spots": len(results["spots"]),
            "occupied": occupied_count,
            "empty": empty_count,
            "occupied_percent": round(100 * occupied_count / len(results["spots"]), 1)
        })
    else:
        results_summary.append({
            "image": image_path.name,
            "status": "NO_RESULTS"
        })

# Print summary
print(f"\n{'='*70}")
print("BATCH TEST SUMMARY")
print(f"{'='*70}\n")

for item in results_summary:
    if item["status"] == "SUCCESS":
        print(f"{item['image']:50s} | Occupied: {item['occupied']:2d}/{item['total_spots']:2d} ({item['occupied_percent']:5.1f}%)")
    else:
        print(f"{item['image']:50s} | {item['status']}")

# Save summary to file
with open(RESULTS_DIR / "batch_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✓ Summary saved to {RESULTS_DIR / 'batch_summary.json'}")

print(f"Results saved to {RESULTS_DIR}/")
