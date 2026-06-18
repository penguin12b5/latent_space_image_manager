"""
Weight sensitivity analysis for LOL-SAM preference.

Runs the same evaluation metrics under alternative weighting schemes
to test whether the LOL-SAM > LOL-FADE conclusion is robust to
the choice of weights (0.45/0.35/0.20 in the original).

Schemes:
  - equal:     1/3 edge, 1/3 smoothness, 1/3 object preservation
  - edge_only: 1.0 edge, 0.0 smoothness, 0.0 object preservation
"""

import csv
import json
import os
import sys
from pathlib import Path

# Ensure scripts/ directory is on path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from evaluate_image_blending import (
    edge_blending_score,
    background_smoothness_score,
    object_preservation_score,
    extract_mask_grabcut,
    resize_to_match,
)

# Repo root is one level up from scripts/
_REPO = Path(__file__).resolve().parent.parent

WEIGHTING_SCHEMES = {
    "original": {"edge": 0.45, "smooth": 0.35, "object": 0.20},
    "equal": {"edge": 1/3, "smooth": 1/3, "object": 1/3},
    "edge_only": {"edge": 1.0, "smooth": 0.0, "object": 0.0},
}

METHOD_PATHS = [
    ("DOD-FADE", str(_REPO / "images_eval" / "dod_fade")),
    ("DOD-SAM",  str(_REPO / "images_eval" / "dod_sam")),
    ("LOL-FADE", str(_REPO / "images_eval" / "lol_fade")),
    ("LOL-SAM",  str(_REPO / "images_eval" / "lol_sam")),
]

IMAGES = [
    "car1.png", "car2.png", "car3.png",
    "cat1.png", "cat2.png", "cat3.png",
    "dog1.png", "dog2.png",
    "horse1.png", "horse2.png", "horse3.png",
]


def compute_overall(edge_score, smooth_score, object_score, weights):
    return (
        weights["edge"] * edge_score +
        weights["smooth"] * smooth_score +
        weights["object"] * object_score
    )


def run_evaluation(scheme_name, weights, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    method_averages = {name: [] for name, _ in METHOD_PATHS}

    for image_name in IMAGES:
        original_path = str(_REPO / "images_eval" / "input" / image_name)
        original = cv2.imread(original_path)
        if original is None:
            print(f"  WARNING: skipping {image_name} (not found)")
            continue

        mask = extract_mask_grabcut(original)
        image_results = {}

        for method_name, method_dir in METHOD_PATHS:
            method_path = os.path.join(method_dir, image_name)
            result = cv2.imread(method_path)
            if result is None:
                print(f"  WARNING: skipping {method_name}/{image_name} (not found)")
                continue

            result = resize_to_match(original, result)
            edge = edge_blending_score(result, mask)
            smooth = background_smoothness_score(result, mask)
            obj = object_preservation_score(original, result, mask)
            overall = compute_overall(edge, smooth, obj, weights)

            image_results[method_name] = {
                "edge_blending": round(edge, 4),
                "background_smoothness": round(smooth, 4),
                "object_preservation": round(obj, 4),
                "overall": round(overall, 4),
            }
            method_averages[method_name].append(overall)

        all_results[image_name] = image_results

    # Compute averages
    averages = {}
    for method_name, scores in method_averages.items():
        if scores:
            averages[method_name] = round(np.mean(scores), 4)

    summary = {
        "scheme": scheme_name,
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "per_image": all_results,
        "averages": averages,
        "ranking": sorted(averages.keys(), key=lambda m: averages[m], reverse=True),
    }

    # Determine winner counts
    wins = {name: 0 for name, _ in METHOD_PATHS}
    for image_name, image_results in all_results.items():
        if not image_results:
            continue
        best = max(image_results.values(), key=lambda r: r["overall"])["overall"]
        for method_name, result in image_results.items():
            if result["overall"] == best:
                wins[method_name] += 1
    summary["wins_per_method"] = wins

    # Save JSON
    json_path = os.path.join(output_dir, f"results_{scheme_name}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV
    csv_path = os.path.join(output_dir, f"results_{scheme_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "method", "edge_blending", "background_smoothness",
                         "object_preservation", "overall"])
        for image_name, image_results in all_results.items():
            for method_name, scores in image_results.items():
                writer.writerow([
                    image_name, method_name,
                    scores["edge_blending"],
                    scores["background_smoothness"],
                    scores["object_preservation"],
                    scores["overall"],
                ])

    return summary


def print_summary(summary):
    print(f"\n{'='*60}")
    print(f"Scheme: {summary['scheme']}")
    print(f"Weights: edge={summary['weights']['edge']:.2f}, "
          f"smooth={summary['weights']['smooth']:.2f}, "
          f"object={summary['weights']['object']:.2f}")
    print(f"{'='*60}")

    print(f"\n  Average overall scores:")
    for method in summary["ranking"]:
        print(f"    {method:12s}: {summary['averages'][method]:.4f}")

    print(f"\n  Per-image wins:")
    for method, count in summary["wins_per_method"].items():
        print(f"    {method:12s}: {count}/{len(IMAGES)}")

    print(f"\n  Ranking: {' > '.join(summary['ranking'])}")

    # Check LOL-SAM vs LOL-FADE
    lol_sam = summary["averages"].get("LOL-SAM", 0)
    lol_fade = summary["averages"].get("LOL-FADE", 0)
    if lol_sam > lol_fade:
        print(f"  LOL-SAM > LOL-FADE confirmed (delta: {lol_sam - lol_fade:.4f})")
    elif lol_sam == lol_fade:
        print(f"  LOL-SAM == LOL-FADE (tied)")
    else:
        print(f"  LOL-SAM < LOL-FADE (REVERSED, delta: {lol_fade - lol_sam:.4f})")


if __name__ == "__main__":
    output_base = str(_REPO / "results" / "weight_sensitivity")
    os.makedirs(output_base, exist_ok=True)

    summaries = {}
    for scheme_name, weights in WEIGHTING_SCHEMES.items():
        print(f"\nRunning evaluation: {scheme_name} ...")
        summary = run_evaluation(scheme_name, weights, output_base)
        summaries[scheme_name] = summary
        print_summary(summary)

    # Final robustness check
    print(f"\n{'='*60}")
    print("ROBUSTNESS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scheme':<12} {'LOL-SAM':>10} {'LOL-FADE':>10} {'SAM>FADE?':>10}")
    print("-" * 44)
    for scheme_name, summary in summaries.items():
        sam = summary["averages"].get("LOL-SAM", 0)
        fade = summary["averages"].get("LOL-FADE", 0)
        holds = "YES" if sam > fade else ("TIE" if sam == fade else "NO")
        print(f"{scheme_name:<12} {sam:>10.4f} {fade:>10.4f} {holds:>10}")
