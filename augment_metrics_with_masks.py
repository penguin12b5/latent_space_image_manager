#!/usr/bin/env python3
import json
from pathlib import Path
import cv2
import numpy as np
import sys

# workspace paths
WS = Path(__file__).resolve().parent
ROOT = Path('images')
METRICS = WS / 'metrics.json'
OUT_TEX = WS / 'quant_table.tex'

# helper: boundary band
def boundary_band(mask, width=4):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    band = dilated - eroded
    return band.astype(bool)

# compute edge energy
def edge_energy(image, mask, width=4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    band = boundary_band(mask, width=width)
    if band.sum()==0:
        return float('nan')
    return float(np.mean(grad[band]))

# background smoothness (mean abs laplacian on background)
def background_smoothness(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg = gray.copy()
    bg[mask>0] = 0
    lap = cv2.Laplacian(bg, cv2.CV_32F)
    hf_energy = np.mean(np.abs(lap[mask==0]))
    return float(hf_energy)

# object preservation: ssim on bounding box
from skimage.metrics import structural_similarity as ssim

def object_preservation(original, result, mask):
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(mask>0)
    if len(ys)==0 or len(xs)==0:
        return float('nan')
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    obj1 = gray_orig[y1:y2+1, x1:x2+1]
    obj2 = gray_res[y1:y2+1, x1:x2+1]
    try:
        return float(ssim(obj1, obj2))
    except Exception:
        return float('nan')

if not METRICS.exists():
    print('metrics.json not found at', METRICS)
    sys.exit(1)

metrics = json.loads(METRICS.read_text())
images = sorted([p.name for p in (ROOT/'input').iterdir() if p.suffix.lower() in ['.png','.jpg']])
methods = ['dod_fade','dod_sam','lol_fade','lol_sam','output']

agg_boundary = {m: [] for m in methods}
agg_bg = {m: [] for m in methods}
agg_obj = {m: [] for m in methods}

for img in images:
    gt_mask_path = ROOT/'masks'/img
    if not gt_mask_path.exists():
        print('Mask missing for', img)
        continue
    mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (mask>127).astype(np.uint8)
    orig = cv2.imread(str(ROOT/'input'/img))
    for m in methods:
        res_path = ROOT/m/img
        if not res_path.exists():
            continue
        res = cv2.imread(str(res_path))
        if res is None or orig is None:
            continue
        # ensure mask matches result shape
        if mask.shape[:2] != res.shape[:2]:
            mask_resized = cv2.resize(mask, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        be = edge_energy(res, mask_resized)
        bg = background_smoothness(res, mask_resized)
        op = object_preservation(orig, res, mask_resized)
        agg_boundary[m].append(be)
        agg_bg[m].append(bg)
        agg_obj[m].append(op)

# aggregate means
out_boundary = {}
out_bg = {}
out_obj = {}
for m in methods:
    vals = [v for v in agg_boundary[m] if not np.isnan(v)]
    out_boundary[m] = float(np.mean(vals)) if len(vals)>0 else None
    vals = [v for v in agg_bg[m] if not np.isnan(v)]
    out_bg[m] = float(np.mean(vals)) if len(vals)>0 else None
    vals = [v for v in agg_obj[m] if not np.isnan(v)]
    out_obj[m] = float(np.mean(vals)) if len(vals)>0 else None

metrics['boundary_energy'] = out_boundary
metrics['background_laplacian'] = out_bg
metrics['object_ssim'] = out_obj

METRICS.write_text(json.dumps(metrics, indent=2))
print('Updated metrics.json with boundary/object/bg values')

# rewrite LaTeX snippet with the new values (FID, LPIPS from existing metrics)
fid = metrics.get('fid',{})
lpips = metrics.get('lpips',{})

lines = []
lines.append('\\begin{tabularx}{\\linewidth}{|l|c|c|c|c|c|}')
lines.append('\\hline')
lines.append('Method & FID & LPIPS (FG) & IoU & Boundary $E$ & Composite $Q$ \\\\')
lines.append('\\hline')
name_map = {'dod_fade':'DOD-FADE','dod_sam':'DOD-SAM','lol_fade':'LOL-FADE','lol_sam':'LOL-SAM','output':'OUTPUT'}
for m in methods:
    fid_str = f"{fid.get(m):.2f}" if fid.get(m) is not None else '--'
    lp_str = f"{lpips.get(m):.4f}" if lpips.get(m) is not None else '--'
    iou_str = '--'
    be = metrics.get('boundary_energy',{}).get(m)
    be_str = f"{be:.2f}" if be is not None else '--'
    comp_str = '--'
    lines.append(f"{name_map.get(m,m)} & {fid_str} & {lp_str} & {iou_str} & {be_str} & {comp_str} \\\\")
lines.append('\\hline')
lines.append('\\end{tabularx}')
OUT_TEX.write_text('\n'.join(lines))
print('Wrote updated LaTeX snippet to', OUT_TEX)
