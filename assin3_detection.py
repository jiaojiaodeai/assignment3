#!/usr/bin/env python3
"""Assignment 3-2: object detection on TWO still images at thresholds 0.3/0.7.

Jetson Nano + jetson-inference (SSD-MobileNet-v2).
Example:
    python3 assin3_detection.py images/image1.jpg images/image2.jpg

Outputs in ./assignment3_results/:
    image1_threshold_0.3.jpg, image1_threshold_0.7.jpg,
    image2_threshold_0.3.jpg, image2_threshold_0.7.jpg,
    detection_details.csv, comparison.csv, results.txt

NOTE: missed/incorrect detections require visual checking; the script does NOT
invent ground-truth labels or automatically claim a prediction is correct.
"""

import argparse
import csv
import gc
from pathlib import Path

# Different jetson-inference releases expose these modules under different names.
try:
    from jetson_inference import detectNet
    from jetson_utils import loadImage, saveImage, cudaDeviceSynchronize
except ImportError:
    from jetson.inference import detectNet
    from jetson.utils import loadImage, saveImage, cudaDeviceSynchronize

THRESHOLDS = (0.3, 0.7)
DETAIL_FIELDS = [
    "image", "threshold", "detection_number", "class_id", "class_name",
    "confidence", "left", "top", "right", "bottom",
    "width", "height", "area", "center_x", "center_y"
]
SUMMARY_FIELDS = [
    "image", "threshold", "number_of_detected_objects",
    "missed_or_incorrect_detections_manual_review"
]


def values_for_detection(net, image_name, threshold, number, detection):
    """Read the fields requested on Lecture 4, slide 40."""
    class_id = int(detection.ClassID)
    center = detection.Center
    try:
        class_name = net.GetClassDesc(class_id)
    except Exception:
        class_name = str(class_id)

    return {
        "image": image_name,
        "threshold": threshold,
        "detection_number": number,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": float(detection.Confidence),
        "left": float(detection.Left),
        "top": float(detection.Top),
        "right": float(detection.Right),
        "bottom": float(detection.Bottom),
        "width": float(detection.Width),
        "height": float(detection.Height),
        "area": float(detection.Area),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
    }


def format_details(record):
    return (
        "    Detection {detection_number}: {class_name} (ClassID={class_id}), "
        "Confidence={confidence:.4f}\n"
        "      Left={left:.2f}, Top={top:.2f}, Right={right:.2f}, "
        "Bottom={bottom:.2f}\n"
        "      Width={width:.2f}, Height={height:.2f}, Area={area:.2f}, "
        "Center=({center_x:.2f}, {center_y:.2f})"
    ).format(**record)


def main():
    parser = argparse.ArgumentParser(
        description="Detect objects in 2 images using confidence thresholds 0.3 and 0.7."
    )
    parser.add_argument("image1", help="path to image 1 (jpg or png)")
    parser.add_argument("image2", help="path to image 2 (jpg or png)")
    parser.add_argument("--output-dir", default="assignment3_results",
                        help="where to save images and CSV results")
    parser.add_argument("--network", default="ssd-mobilenet-v2",
                        help="pretrained detectNet name (default: ssd-mobilenet-v2)")
    args = parser.parse_args()

    image_paths = [Path(args.image1).expanduser(), Path(args.image2).expanduser()]
    for image_path in image_paths:
        if not image_path.is_file():
            parser.error("input image not found: {}".format(image_path))

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    details = []
    comparisons = []
    log = ["Assignment 3-2 | SSD-MobileNet-v2 Object Detection"]

    # Create one model at a time (useful on RAM-limited Jetson Nano 2GB).
    for threshold in THRESHOLDS:
        print("\n===== Loading {} with threshold {} =====".format(args.network, threshold))
        net = detectNet(args.network, threshold=threshold)

        for image_num, image_path in enumerate(image_paths, 1):
            # Re-load the original: net.Detect overlays bounding boxes in-place.
            img = loadImage(str(image_path))
            if img is None:
                raise RuntimeError("Unable to load: {}".format(image_path))

            detections = net.Detect(img, overlay="box,labels,conf")
            file_out = out_dir / "image{}_threshold_{:.1f}.jpg".format(
                image_num, threshold)
            cudaDeviceSynchronize()
            saveImage(str(file_out), img)

            header = "Image {} ({}) | threshold={} | detected={} | output={}".format(
                image_num, image_path, threshold, len(detections), file_out)
            print("\n" + header)
            log.append("\n" + header)

            comparisons.append({
                "image": "Image {}".format(image_num),
                "threshold": threshold,
                "number_of_detected_objects": len(detections),
                "missed_or_incorrect_detections_manual_review": "FILL AFTER VIEWING IMAGE"
            })

            if not detections:
                print("    No objects detected at this threshold.")
                log.append("    No objects detected at this threshold.")

            for det_number, det in enumerate(detections, 1):
                record = values_for_detection(
                    net, "Image {}".format(image_num), threshold, det_number, det)
                details.append(record)
                line = format_details(record)
                print(line)
                log.append(line)

            del img

        del net
        gc.collect()

    with (out_dir / "detection_details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(details)

    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(comparisons)

    with (out_dir / "results.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")

    print("\nDone! Files saved under: {}".format(out_dir))
    print("IMPORTANT: fill the missed/incorrect column in comparison.csv manually")
    print("after examining the original and annotated images.")


if __name__ == "__main__":
    main()
