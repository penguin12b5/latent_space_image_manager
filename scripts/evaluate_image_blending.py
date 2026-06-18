import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# Repo root is one level up from scripts/
_REPO = Path(__file__).resolve().parent.parent


def extract_mask_grabcut(image, border_ratio=0.05, iterations=5):
    """
    Extract foreground mask using GrabCut.

    Parameters
    ----------
    image : np.ndarray
        BGR image loaded by cv2.imread().
    border_ratio : float
        Percentage of image treated as background border.
    iterations : int
        GrabCut iterations.

    Returns
    -------
    mask : np.uint8
        Binary mask:
        1 = foreground
        0 = background
    """

    h, w = image.shape[:2]

    margin_x = int(w * border_ratio)
    margin_y = int(h * border_ratio)

    rect = (
        margin_x,
        margin_y,
        w - 2 * margin_x,
        h - 2 * margin_y,
    )

    grabcut_mask = np.zeros((h, w), np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        grabcut_mask,
        rect,
        bgd_model,
        fgd_model,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )

    mask = np.where(
        (grabcut_mask == cv2.GC_FGD) |
        (grabcut_mask == cv2.GC_PR_FGD),
        1,
        0,
    ).astype(np.uint8)

    return mask


def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)
    return mask


def boundary_band(mask, width=5):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1)
    )

    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)

    band = dilated - eroded
    return band.astype(bool)


def resize_to_match(reference, image):
    if image.shape[:2] != reference.shape[:2]:
        return cv2.resize(
            image,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return image


def edge_blending_score(image, mask):
    """
    Lower boundary gradient jump -> better blending
    Returns score in [0,1]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    grad = np.sqrt(gx**2 + gy**2)

    band = boundary_band(mask, width=4)

    edge_energy = grad[band].mean()

    score = np.exp(-edge_energy / 50.0)
    return float(score)


def background_smoothness_score(image, mask):
    """
    Measures reduction of high-frequency energy
    Higher score => smoother background
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    bg = gray.copy()
    bg[mask > 0] = 0

    lap = cv2.Laplacian(bg, cv2.CV_32F)

    hf_energy = np.mean(np.abs(lap[mask == 0]))

    score = np.exp(-hf_energy / 30.0)

    return float(score)


def object_preservation_score(original, result, mask):
    """
    SSIM over foreground object only
    """
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    ys, xs = np.where(mask > 0)

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    obj1 = gray_orig[y1:y2 + 1, x1:x2 + 1]
    obj2 = gray_res[y1:y2 + 1, x1:x2 + 1]

    score = ssim(obj1, obj2)

    return float(score)


def evaluate_method(original, result, mask):
    result = resize_to_match(original, result)

    edge_score = edge_blending_score(result, mask)
    smooth_score = background_smoothness_score(result, mask)
    object_score = object_preservation_score(
        original, result, mask
    )

    total = (
        0.45 * edge_score +
        0.35 * smooth_score +
        0.20 * object_score
    )

    return {
        "edge_blending": edge_score,
        "background_smoothness": smooth_score,
        "object_preservation": object_score,
        "overall": total,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-mask",
        action="store_true",
        help="Save GrabCut masks to results/masks (default: disabled)",
    )
    parser.add_argument(
        "--result-file",
        default=str(
            _REPO / "results" / "evaluate_image_blending_result.json"
        ),
        help="Output JSON file for evaluation results",
    )

    args = parser.parse_args()

    method_paths = [
        ("Method 1", str(_REPO / "images_eval" / "dod_fade")),
        ("Method 2", str(_REPO / "images_eval" / "dod_sam")),
        ("Method 3", str(_REPO / "images_eval" / "lol_fade")),
        ("Method 4", str(_REPO / "images_eval" / "lol_sam")),
    ]

    images = [
        "car1.png", "car2.png", "car3.png",
        "cat1.png", "cat2.png", "cat3.png",
        "dog1.png", "dog2.png",
        "horse1.png", "horse2.png", "horse3.png",
    ]

    all_results = {}

    for image_name in images:

        original_path = str(
            _REPO / "images_eval" / "input" / image_name
        )

        original = cv2.imread(original_path)

        if original is None:
            raise FileNotFoundError(
                f"Original image not found: {original_path}"
            )

        mask = extract_mask_grabcut(original)

        # Optional mask output
        if args.output_mask:
            out_mask_dir = _REPO / "results" / "masks"
            out_mask_dir.mkdir(parents=True, exist_ok=True)

            mask_path = out_mask_dir / image_name

            cv2.imwrite(
                str(mask_path),
                (mask * 255).astype(np.uint8),
            )

            print(f"Saved GrabCut mask: {mask_path}")

        print(f"\nEvaluating image: {image_name}")

        scores = {}

        for name, method_dir in method_paths:

            method_path = os.path.join(
                method_dir,
                image_name,
            )
            print(f"Loading method result: {method_path}")

            result = cv2.imread(method_path)

            if result is None:
                raise FileNotFoundError(
                    f"Method image not found: {method_path}"
                )

            scores[name] = evaluate_method(
                original,
                result,
                mask,
            )

        for name, result in scores.items():
            print(f"\n{name}")

            for k, v in result.items():
                print(f"{k:25s}: {v:.4f}")

        best_score = max(
            result["overall"]
            for result in scores.values()
        )

        winners = [
            name
            for name, result in scores.items()
            if result["overall"] == best_score
        ]

        if len(winners) == 1:
            print(f"\nWinner: {winners[0]}")
        else:
            print(f"\nTie: {', '.join(winners)}")

        all_results[image_name] = {
            "scores": scores,
            "winners": winners,
            "best_score": float(best_score),
        }

    # Save JSON results
    result_path = Path(args.result_file)
    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(result_path, "w") as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    print(f"\nSaved results to: {result_path}")
    