"""
Generate ablation data for the stylization effect claim (Contribution 3).

Produces:
  1. Mask-scale sweep: LOL-SAM at alpha in {0.0, 0.5, 1.0, 1.5, 2.0}
  2. Latent histogram stats: per-alpha activation statistics before decode
  3. Pixel-space alpha>1 control: RGB scaling comparison
  4. Filter baselines: unsharp mask, bilateral filter, CLAHE

Outputs:
  images/output/ablation/<image>_<variant>.png
  results/stylization_ablation_results.json
  results/stylization_ablation_summary.csv
"""

import os
import sys
import json
import csv
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

# Repo root is one level up from scripts/
_REPO = Path(__file__).resolve().parent.parent

# Keep scripts/ on sys.path for any sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from diffusers import AutoencoderKL
from segment_anything import sam_model_registry, SamPredictor
from skimage.metrics import structural_similarity as ssim

import torchvision


# ---------------------------------------------------------------------------
# Core pipeline components (extracted from process_image.py to avoid its
# module-level model instantiation side effects)
# ---------------------------------------------------------------------------

class ImageProcessor:
    def __init__(self, detection_threshold=0.8):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.detection_threshold = detection_threshold
        self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        ).to(self.device).eval()
        self.vae = AutoencoderKL.from_pretrained(str(_REPO / "models")).to(self.device).eval()

    def find_subjects(self, image, max_width=1000, max_height=1000):
        image_tensor = transforms.ToTensor()(image).to(self.device)
        with torch.no_grad():
            outputs = self.detection_model([image_tensor])[0]
        boxes, scores = outputs["boxes"], outputs["scores"]
        keep = [i for i, s in enumerate(scores) if s > self.detection_threshold]
        subjects = []
        for i in keep:
            x1, y1, x2, y2 = map(int, boxes[i].tolist())
            w, h = x2 - x1, y2 - y1
            if w <= max_width and h <= max_height:
                subjects.append((image.crop((x1, y1, x2, y2)), (x1, y1)))
            else:
                hs = (w + max_width - 1) // max_width
                vs = (h + max_height - 1) // max_height
                tw, th = w / hs, h / vs
                for r in range(vs):
                    for c in range(hs):
                        l = int(x1 + c * tw)
                        u = int(y1 + r * th)
                        ri = int(min(l + tw, x2))
                        lo = int(min(u + th, y2))
                        subjects.append((image.crop((l, u, ri, lo)), (l, u)))
        return subjects or None

    def encode(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        t = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.vae.encode(t).latent_dist.sample()

    def decode(self, latent):
        with torch.no_grad():
            return self.vae.decode(latent).sample

    def resize_image(self, image, scale=1.0):
        w, h = image.size
        return image.resize((int(w * scale), int(h * scale)))


def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).cpu()
    return to_pil_image((tensor.clamp(-1, 1) + 1) / 2)


def load_image(file_path):
    return Image.open(file_path).convert("RGB")


def get_sam_mask(image, box, sam_checkpoint, sam_model_type="vit_h", device=None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if sam_model_type == "vit_h" and sam_checkpoint and "vit_b" in sam_checkpoint:
        sam_model_type = "vit_b"
    x1, y1, x2, y2 = map(int, box)
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(np.array(image))
    masks, _, _ = predictor.predict(
        box=np.array([[x1, y1, x2, y2]]), multimask_output=False,
    )
    mask = masks[0].astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask, mode="L")
    if mask_pil.size != (x2 - x1, y2 - y1):
        mask_pil = mask_pil.crop((x1, y1, x2, y2))
    return mask_pil


# ---------------------------------------------------------------------------
# Evaluation helpers (extracted from evaluate_image_blending.py)
# ---------------------------------------------------------------------------

def extract_mask_grabcut(image, border_ratio=0.05, iterations=5):
    h, w = image.shape[:2]
    mx, my = int(w * border_ratio), int(h * border_ratio)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    gc_mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, gc_mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    return np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)


def boundary_band(mask, width=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return (cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)).astype(bool)


def edge_blending_score(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    band = boundary_band(mask, width=4)
    return float(np.exp(-grad[band].mean() / 50.0))


def background_smoothness_score(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg = gray.copy()
    bg[mask > 0] = 0
    lap = cv2.Laplacian(bg, cv2.CV_32F)
    return float(np.exp(-np.mean(np.abs(lap[mask == 0])) / 30.0))


def object_preservation_score(original, result, mask):
    g1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(mask > 0)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    return float(ssim(g1[y1:y2 + 1, x1:x2 + 1], g2[y1:y2 + 1, x1:x2 + 1]))

ALPHA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]
PIXEL_ALPHA_VALUES = [1.5, 2.0]
SCALE = 0.25

IMAGES = [
    "car1.png", "car2.png", "car3.png",
    "cat1.png", "cat2.png", "cat3.png",
    "dog1.png", "dog2.png",
    "horse1.png", "horse2.png", "horse3.png",
]

OUTPUT_DIR = str(_REPO / "images" / "output" / "ablation")
RESULTS_DIR = str(_REPO / "results")


def latent_stats(z):
    vals = z.float().flatten()
    mean = vals.mean().item()
    std = vals.std().item()
    skew = ((vals - mean) ** 3).mean().item() / max(std ** 3, 1e-12)
    return {
        "mean": mean,
        "std": std,
        "skewness": skew,
        "min": vals.min().item(),
        "max": vals.max().item(),
        "frac_outside_1": (vals.abs() > 1.0).float().mean().item(),
        "frac_outside_3": (vals.abs() > 3.0).float().mean().item(),
    }


def run_lol_sam_with_alpha(image_path, alpha, processor):
    image = load_image(image_path)
    subjects = processor.find_subjects(image)
    if not subjects:
        return None, None

    encoded_bg = processor.encode(processor.resize_image(image, SCALE))
    image_latent = F.interpolate(
        encoded_bg, scale_factor=1 / SCALE,
        mode="bilinear", align_corners=False,
    )
    merged_latent = image_latent.clone()

    sam_checkpoint = os.environ.get("SAM_CHECKPOINT", "models/sam_vit_h_4b8939.pth")

    for subject_image, (x1, y1) in subjects:
        encoded_subject = processor.encode(subject_image)
        _, _, h, w = encoded_subject.shape
        top = y1 // 8
        left = x1 // 8

        try:
            mask_pil = get_sam_mask(
                image,
                (x1, y1, x1 + subject_image.width, y1 + subject_image.height),
                sam_checkpoint,
            )
        except Exception:
            print(f"  SAM failed for subject at ({x1},{y1}), skipping")
            continue

        mask_tensor = torch.from_numpy(np.array(mask_pil))[None, None, ...].to(
            device=merged_latent.device, dtype=merged_latent.dtype
        )
        mask_tensor = F.interpolate(
            mask_tensor, size=(h, w), mode="bilinear", align_corners=False,
        )
        mask_tensor = mask_tensor / 255.0 * alpha

        base_patch = merged_latent[:, :, top:top + h, left:left + w]
        blended = base_patch * (1 - mask_tensor) + encoded_subject * mask_tensor
        merged_latent[:, :, top:top + h, left:left + w] = blended

    stats = latent_stats(merged_latent)
    output_image = tensor_to_pil(processor.decode(merged_latent))
    return output_image, stats


def run_pixel_alpha_control(image_path, alpha, processor):
    image = load_image(image_path)
    subjects = processor.find_subjects(image)
    if not subjects:
        return None

    decoded_bg = processor.decode(
        processor.encode(processor.resize_image(image, SCALE))
    )
    bg_image = processor.resize_image(tensor_to_pil(decoded_bg), 1 / SCALE)

    sam_checkpoint = os.environ.get("SAM_CHECKPOINT", "models/sam_vit_h_4b8939.pth")

    for subject_image, (x1, y1) in subjects:
        w, h = subject_image.size
        try:
            mask_pil = get_sam_mask(
                image, (x1, y1, x1 + w, y1 + h), sam_checkpoint,
            )
        except Exception:
            print(f"  SAM failed for pixel control at ({x1},{y1}), skipping")
            continue

        subject_arr = np.array(subject_image).astype(np.float32)
        scaled = np.clip(subject_arr * alpha, 0, 255).astype(np.uint8)
        scaled_pil = Image.fromarray(scaled)

        mask_resized = mask_pil.resize((w, h), Image.BILINEAR)
        bg_image.paste(scaled_pil, (x1, y1), mask_resized)

    return bg_image


def run_filter_baselines(image_path):
    image = cv2.imread(image_path)
    results = {}

    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    results["unsharp"] = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    results["bilateral"] = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge([l_eq, a_ch, b_ch])
    results["clahe"] = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return results


def stylization_strength(original_bgr, result_bgr, mask):
    orig_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    res_gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    orig_lap = np.abs(cv2.Laplacian(orig_gray, cv2.CV_32F))
    res_lap = np.abs(cv2.Laplacian(res_gray, cv2.CV_32F))

    fg = mask > 0
    if fg.sum() == 0:
        return 0.0

    orig_energy = orig_lap[fg].mean()
    res_energy = res_lap[fg].mean()

    if orig_energy < 1e-6:
        return 0.0
    return float(res_energy / orig_energy)


def evaluate_output(original_bgr, result_bgr, mask):
    if result_bgr.shape[:2] != original_bgr.shape[:2]:
        result_bgr = cv2.resize(
            result_bgr, (original_bgr.shape[1], original_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return {
        "edge_blending": edge_blending_score(result_bgr, mask),
        "background_smoothness": background_smoothness_score(result_bgr, mask),
        "object_preservation": object_preservation_score(original_bgr, result_bgr, mask),
        "stylization_strength": stylization_strength(original_bgr, result_bgr, mask),
    }


def pil_to_bgr(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def save_bgr(bgr, path):
    cv2.imwrite(path, bgr)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if "SAM_CHECKPOINT" not in os.environ:
        os.environ["SAM_CHECKPOINT"] = str(_REPO / "models" / "sam_vit_h_4b8939.pth")

    print("Loading models...")
    processor = ImageProcessor()

    all_results = {}

    for img_name in IMAGES:
        stem = os.path.splitext(img_name)[0]
        image_path = str(_REPO / "images" / "input" / img_name)

        if not os.path.exists(image_path):
            print(f"Skipping {img_name}: not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {img_name}")
        print(f"{'='*60}")

        original_bgr = cv2.imread(image_path)
        mask = extract_mask_grabcut(original_bgr)

        image_results = {"image": img_name, "ablations": {}}

        # --- 1. Mask-scale sweep ---
        for alpha in ALPHA_VALUES:
            label = f"latent_alpha_{alpha:.1f}"
            print(f"  LOL-SAM alpha={alpha:.1f} ...", end=" ", flush=True)

            output_pil, stats = run_lol_sam_with_alpha(image_path, alpha, processor)
            if output_pil is None:
                print("no subjects detected")
                continue

            out_path = os.path.join(OUTPUT_DIR, f"{stem}_alpha_{alpha:.1f}.png")
            output_pil.save(out_path)

            result_bgr = pil_to_bgr(output_pil)
            if result_bgr.shape[:2] != original_bgr.shape[:2]:
                result_bgr = cv2.resize(
                    result_bgr, (original_bgr.shape[1], original_bgr.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            metrics = evaluate_output(original_bgr, result_bgr, mask)
            metrics["latent_stats"] = stats

            image_results["ablations"][label] = metrics
            print(f"done (stylization={metrics['stylization_strength']:.3f})")

        # --- 2. Pixel-space alpha>1 control ---
        for alpha in PIXEL_ALPHA_VALUES:
            label = f"pixel_alpha_{alpha:.1f}"
            print(f"  Pixel-space alpha={alpha:.1f} ...", end=" ", flush=True)

            output_pil = run_pixel_alpha_control(image_path, alpha, processor)
            if output_pil is None:
                print("no subjects detected")
                continue

            out_path = os.path.join(OUTPUT_DIR, f"{stem}_pixel_alpha_{alpha:.1f}.png")
            output_pil.save(out_path)

            result_bgr = pil_to_bgr(output_pil)
            if result_bgr.shape[:2] != original_bgr.shape[:2]:
                result_bgr = cv2.resize(
                    result_bgr, (original_bgr.shape[1], original_bgr.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            metrics = evaluate_output(original_bgr, result_bgr, mask)

            image_results["ablations"][label] = metrics
            print(f"done (stylization={metrics['stylization_strength']:.3f})")

        # --- 3. Filter baselines ---
        filter_outputs = run_filter_baselines(image_path)
        for filter_name, filter_bgr in filter_outputs.items():
            label = f"filter_{filter_name}"
            print(f"  Filter: {filter_name} ...", end=" ", flush=True)

            out_path = os.path.join(OUTPUT_DIR, f"{stem}_{filter_name}.png")
            save_bgr(filter_bgr, out_path)

            metrics = evaluate_output(original_bgr, filter_bgr, mask)
            image_results["ablations"][label] = metrics
            print(f"done (stylization={metrics['stylization_strength']:.3f})")

        all_results[img_name] = image_results

    # --- Save JSON results ---
    json_path = os.path.join(RESULTS_DIR, "stylization_ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {json_path}")

    # --- Save CSV summary ---
    csv_path = os.path.join(RESULTS_DIR, "stylization_ablation_summary.csv")
    write_summary_csv(all_results, csv_path)
    print(f"Summary CSV saved to {csv_path}")

    # --- Print summary table ---
    print_summary_table(all_results)


def write_summary_csv(all_results, csv_path):
    fieldnames = [
        "image", "method",
        "edge_blending", "background_smoothness", "object_preservation",
        "composite_score", "stylization_strength",
        "latent_mean", "latent_std", "latent_skewness",
        "latent_min", "latent_max",
        "frac_outside_1", "frac_outside_3",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for img_name, img_data in all_results.items():
            for method, metrics in img_data["ablations"].items():
                row = {
                    "image": img_name,
                    "method": method,
                    "edge_blending": f"{metrics['edge_blending']:.4f}",
                    "background_smoothness": f"{metrics['background_smoothness']:.4f}",
                    "object_preservation": f"{metrics['object_preservation']:.4f}",
                    "composite_score": f"{0.5 * metrics['edge_blending'] + 0.3 * metrics['background_smoothness'] + 0.2 * metrics['object_preservation']:.4f}",
                    "stylization_strength": f"{metrics['stylization_strength']:.4f}",
                }

                stats = metrics.get("latent_stats")
                if stats:
                    row.update({
                        "latent_mean": f"{stats['mean']:.4f}",
                        "latent_std": f"{stats['std']:.4f}",
                        "latent_skewness": f"{stats['skewness']:.4f}",
                        "latent_min": f"{stats['min']:.4f}",
                        "latent_max": f"{stats['max']:.4f}",
                        "frac_outside_1": f"{stats['frac_outside_1']:.4f}",
                        "frac_outside_3": f"{stats['frac_outside_3']:.4f}",
                    })

                writer.writerow(row)


def print_summary_table(all_results):
    print(f"\n{'='*100}")
    print("AGGREGATED SUMMARY (mean across images)")
    print(f"{'='*100}")

    method_agg = {}

    for img_data in all_results.values():
        for method, metrics in img_data["ablations"].items():
            if method not in method_agg:
                method_agg[method] = {
                    "edge": [], "smooth": [], "obj": [],
                    "stylization": [], "latent_std": [], "frac_out_1": [],
                }

            method_agg[method]["edge"].append(metrics["edge_blending"])
            method_agg[method]["smooth"].append(metrics["background_smoothness"])
            method_agg[method]["obj"].append(metrics["object_preservation"])
            method_agg[method]["stylization"].append(metrics["stylization_strength"])

            stats = metrics.get("latent_stats")
            if stats:
                method_agg[method]["latent_std"].append(stats["std"])
                method_agg[method]["frac_out_1"].append(stats["frac_outside_1"])

    header = f"{'Method':<25} {'Edge':>8} {'Smooth':>8} {'ObjPres':>8} {'Composite':>10} {'Styliz':>8} {'LatStd':>8} {'%Out1':>8}"
    print(header)
    print("-" * len(header))

    for method in sorted(method_agg.keys()):
        agg = method_agg[method]
        e = np.mean(agg["edge"])
        s = np.mean(agg["smooth"])
        o = np.mean(agg["obj"])
        comp = 0.5 * e + 0.3 * s + 0.2 * o
        sty = np.mean(agg["stylization"])
        lstd = np.mean(agg["latent_std"]) if agg["latent_std"] else float("nan")
        fout = np.mean(agg["frac_out_1"]) if agg["frac_out_1"] else float("nan")

        print(f"{method:<25} {e:>8.4f} {s:>8.4f} {o:>8.4f} {comp:>10.4f} {sty:>8.3f} {lstd:>8.4f} {fout:>8.4f}")


if __name__ == "__main__":
    main()
