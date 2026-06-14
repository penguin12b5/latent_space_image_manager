#!/usr/bin/env python3
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision

ROOT = Path('images')
OUT = Path('results')
METRICS = OUT / 'metrics.json'

def binarize_mask(arr):
    return (arr > 127).astype(np.uint8)

def paste_roi_mask(full_shape, roi_mask, x1, y1):
    H, W = full_shape
    full = np.zeros((H, W), dtype=np.uint8)
    roi_mask = np.asarray(roi_mask).astype(np.uint8)
    h, w = roi_mask.shape
    # clamp destination coordinates to image bounds
    x1c = max(0, int(x1))
    y1c = max(0, int(y1))
    x2c = min(W, x1c + w)
    y2c = min(H, y1c + h)
    dest_h = y2c - y1c
    dest_w = x2c - x1c
    if dest_h <= 0 or dest_w <= 0:
        return full
    # source crop within roi_mask
    src_y0 = max(0, y1c - int(y1))
    src_x0 = max(0, x1c - int(x1))
    src_y1 = src_y0 + dest_h
    src_x1 = src_x0 + dest_w
    src_crop = roi_mask[src_y0:src_y1, src_x0:src_x1]
    # if source crop doesn't match destination shape, resize nearest
    if src_crop.shape != (dest_h, dest_w):
        from PIL import Image as PILImage
        src_crop = np.array(PILImage.fromarray(roi_mask).resize((dest_w, dest_h), resample=PILImage.NEAREST))
    full[y1c:y2c, x1c:x2c] = src_crop
    return full

def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)

def get_fade_mask(w, h, fade_px):
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    dist_left = x
    dist_right = (w - 1) - x
    dist_top = y
    dist_bottom = (h - 1) - y
    dist_edge = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom)).astype(np.float32)
    alpha = np.clip(dist_edge / max(1.0, float(fade_px)), 0.0, 1.0)
    alpha = np.power(alpha, 1.5)
    mask = (alpha * 255.0).astype(np.uint8)
    return mask

def main():
    device = torch.device('cpu')
    detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    detection_model.to(device).eval()

    # SAM
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except Exception as e:
        print('segment_anything not available:', e)
        sam_model_registry = None

    images = sorted([p.name for p in (ROOT / 'input').iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg']])

    per_image = {}
    sam_ious = []
    fade_ious = []

    for img_name in images:
        img_path = ROOT / 'input' / img_name
        img_pil = Image.open(img_path).convert('RGB')
        W, H = img_pil.size

        # detection
        transform = transforms.ToTensor()
        img_t = transform(img_pil).to(device)
        with torch.no_grad():
            out = detection_model([img_t])[0]
        scores = out['scores'].cpu().numpy()
        boxes = out['boxes'].cpu().numpy()
        keep = np.where(scores > 0.8)[0]
        if len(keep) == 0:
            print('no detections for', img_name)
            continue
        idx = keep[0]
        x1, y1, x2, y2 = boxes[idx].astype(int)
        w = x2 - x1
        h = y2 - y1

        # SAM mask
        full_sam = None
        if sam_model_registry is not None:
            sam_checkpoint = os.environ.get('SAM_CHECKPOINT', None)
            if sam_checkpoint is None:
                os.environ['SAM_CHECKPOINT'] = 'models/sam_vit_h_4b8939.pth'
                sam_checkpoint = os.environ['SAM_CHECKPOINT']
            try:
                sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
                sam.to(device='cpu')
                predictor = SamPredictor(sam)
                predictor.set_image(np.array(img_pil))
                masks, scores_sam, logits = predictor.predict(box=np.array([[x1, y1, x2, y2]]), multimask_output=False)
                mask_full = (masks[0].astype(np.uint8) * 255)
                # SAM may return a mask at the full-image resolution or at the ROI size.
                if mask_full.shape == (h, w):
                    mask_roi = mask_full
                elif mask_full.shape == (H, W):
                    mask_roi = mask_full[y1:y1+h, x1:x1+w]
                else:
                    # Unexpected shape: resize to ROI
                    from PIL import Image as PILImage
                    mask_roi = np.array(PILImage.fromarray(mask_full).resize((w, h), resample=PILImage.NEAREST))
                full_sam = paste_roi_mask((H, W), mask_roi, x1, y1)
            except Exception as e:
                print('SAM failed for', img_name, e)

        # Fade mask
        fade_mask = get_fade_mask(w, h, int(0.08 * min(w, h)))
        fade_bin = binarize_mask(fade_mask)
        full_fade = paste_roi_mask((H, W), fade_bin, x1, y1)

        # GT
        gt_path = ROOT / 'masks' / img_name
        if not gt_path.exists():
            print('GT missing for', img_name)
            continue
        gt = np.array(Image.open(gt_path).convert('L'))
        gt_bin = binarize_mask(gt)

        res = {}
        if full_sam is not None:
            val = iou(gt_bin, (full_sam>0).astype(np.uint8))
            res['sam_iou'] = val
            sam_ious.append(val)
        else:
            res['sam_iou'] = None

        val2 = iou(gt_bin, (full_fade>0).astype(np.uint8))
        res['fade_iou'] = val2
        fade_ious.append(val2)

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
