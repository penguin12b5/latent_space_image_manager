#!/usr/bin/env python3
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
from process_image import ImageProcessor, load_image

ROOT = Path('images')
OUT = Path('results')
METRICS = OUT / 'metrics.json'

def binarize_mask(arr):
    return (arr > 127).astype(np.uint8)

def paste_roi_mask(full_shape, roi_mask, x1, y1):
    H, W = full_shape
    full = np.zeros((H, W), dtype=np.uint8)
    h, w = roi_mask.shape
    full[y1:y1+h, x1:x1+w] = roi_mask
    return full

def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)

def main():
    p = ImageProcessor()
    images = sorted([p.name for p in (ROOT / 'input').iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg']])

    per_image = {}
    sam_ious = []
    fade_ious = []

    for img_name in images:
        img_path = ROOT / 'input' / img_name
        img = load_image(str(img_path))
        H, W = img.size[1], img.size[0]

        subjects = p.find_subjects(img)
        if not subjects:
            print('No subjects for', img_name)
            continue

        # use primary subject
        subject_image, (x1, y1) = subjects[0]
        w, h = subject_image.size

        # SAM mask (roi crop)
        try:
            sam_checkpoint = os.environ.get('SAM_CHECKPOINT', None)
            mask_roi = p.get_sam_mask(img, (x1, y1, x1 + w, y1 + h), sam_checkpoint)
            mask_arr = np.array(mask_roi)
            mask_bin = binarize_mask(mask_arr)
            full_sam = paste_roi_mask((H, W), mask_bin, x1, y1)
        except Exception as e:
            print('SAM mask failed for', img_name, e)
            full_sam = None

        # Fade mask (heuristic) - use get_fade_mask from process_image
        try:
            fade_pil = p.get_fade_mask(w, h, int(0.08 * min(w, h)))
            fade_arr = np.array(fade_pil)
            fade_bin = binarize_mask(fade_arr)
            full_fade = paste_roi_mask((H, W), fade_bin, x1, y1)
        except Exception as e:
            print('Fade mask failed for', img_name, e)
            full_fade = None

        # load grabcut mask
        gt_path = ROOT / 'masks' / img_name
        if not gt_path.exists():
            print('GT mask missing for', img_name)
            continue
        gt = np.array(Image.open(gt_path).convert('L'))
        gt_bin = binarize_mask(gt)

        res = {}
        if full_sam is not None:
            val = iou(gt_bin, full_sam)
            res['sam_iou'] = val
            sam_ious.append(val)
        else:
            res['sam_iou'] = None

        if full_fade is not None:
            val = iou(gt_bin, full_fade)
            res['fade_iou'] = val
            fade_ious.append(val)
        else:
            res['fade_iou'] = None

        per_image[img_name] = res

    agg = {
        'dod_sam': float(np.mean(sam_ious)) if len(sam_ious)>0 else None,
        'lol_sam': float(np.mean(sam_ious)) if len(sam_ious)>0 else None,
        'dod_fade': float(np.mean(fade_ious)) if len(fade_ious)>0 else None,
        'lol_fade': float(np.mean(fade_ious)) if len(fade_ious)>0 else None,
    }

    out = {}
    if METRICS.exists():
        try:
            out = json.loads(METRICS.read_text())
        except Exception:
            out = {}

    out['iou'] = agg
    out['per_image_iou'] = per_image
    METRICS.write_text(json.dumps(out, indent=2))
    print('Wrote IoU results to', METRICS)

if __name__ == '__main__':
    main()
