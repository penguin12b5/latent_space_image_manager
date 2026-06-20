#!/usr/bin/env python3
import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm
import scipy.linalg

# Config
# Repo root is one level up from scripts/
_REPO = Path(__file__).resolve().parent.parent
ROOT = _REPO / 'images'
METHOD_FOLDERS = ['dod_fade','dod_sam','lol_fade','lol_sam','dod_fade','output']
METHODS = ['DOD-FADE','DOD-SAM','LOL-FADE','LOL-SAM','DOD-FADE-dup','OUTPUT']
OUTPUT_FOLDER = ROOT / 'output'
INPUT_FOLDER = ROOT / 'input'
SCALE_SUFFIX = '_0.25'
OUT_DIR = _REPO / 'results'
METRICS_OUT = OUT_DIR / 'metrics.json'
PER_IMAGE_CSV = OUT_DIR / 'per_image_metrics.csv'
LATEX_SNIPPET = OUT_DIR / 'quant_table.tex'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    mps = getattr(torch.backends, 'mps', None)
    if mps and mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
print('Using device:', device)

# Utilities
def list_input_images():
    imgs = sorted([p.name for p in INPUT_FOLDER.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg']])
    return imgs

def method_image_path(method, img_name):
    stem = Path(img_name).stem
    suffix = Path(img_name).suffix
    return OUTPUT_FOLDER / method / f'{stem}{SCALE_SUFFIX}{suffix}'

# Feature extractor (ResNet50 avgpool features)
class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        f = self.backbone(x)
        return f.view(f.size(0), -1)

preprocess_224 = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

preprocess_lpips = T.Compose([
    T.Resize((256,256)),
    T.ToTensor()
])

# Load images to tensor
def load_image_tensor(path, for_lpips=False):
    img = Image.open(path).convert('RGB')
    if for_lpips:
        return preprocess_lpips(img).unsqueeze(0)
    return preprocess_224(img).unsqueeze(0)

# Compute FID between two lists of image paths using ResNet50 features
def compute_fid(paths1, paths2, batch_size=8):
    if len(paths1)==0 or len(paths2)==0:
        return None
    model = ResNetFeatureExtractor().to(device).eval()
    def get_acts(paths):
        acts = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            imgs = torch.cat([load_image_tensor(p) for p in batch], dim=0).to(device)
            with torch.no_grad():
                a = model(imgs).cpu().numpy()
            acts.append(a)
        return np.concatenate(acts, axis=0)
    a1 = get_acts(paths1)
    a2 = get_acts(paths2)
    mu1, sigma1 = np.mean(a1, axis=0), np.cov(a1, rowvar=False)
    mu2, sigma2 = np.mean(a2, axis=0), np.cov(a2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return float(fid)

# LPIPS (try lpips package, else use VGG-based perceptual L2)
def compute_lpips_pairs(pairs):
    results = {}
    # Prefer lpips package when available
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='vgg').to(device)
        use_lpips = True
    except Exception:
        loss_fn = None
        use_lpips = False

    # Fallback: VGG-based perceptual distance
    vgg = None
    layer_idx = 16
    if not use_lpips:
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()

    for name, (p_ref, p_cmp) in tqdm(pairs.items(), desc='LPIPS'):
        try:
            img0 = load_image_tensor(p_ref, for_lpips=use_lpips).to(device)
            img1 = load_image_tensor(p_cmp, for_lpips=use_lpips).to(device)
            if use_lpips and loss_fn is not None:
                a = T.functional.resize(img0, (256,256))
                b = T.functional.resize(img1, (256,256))
                a = (a * 2.0 - 1.0)
                b = (b * 2.0 - 1.0)
                with torch.no_grad():
                    val = loss_fn(a, b).mean().item()
            else:
                def feat(x):
                    x = T.functional.resize(x, (224,224))
                    with torch.no_grad():
                        f = vgg[:layer_idx](x)
                    return f.view(f.size(0), -1)
                f0 = feat(img0)
                f1 = feat(img1)
                val = torch.nn.functional.mse_loss(f0, f1).item()
            results[name] = float(val)
        except Exception as e:
            print('LPIPS error for', name, e)
            results[name] = None
    return results

# IoU and Boundary E require masks; try detect masks in ROOT/gt or *_mask.png
def find_mask_for(image_name):
    # look for same name with _mask or in gt folder
    candidates = [ROOT / 'masks' / image_name, ROOT / 'gt' / image_name]
    base = Path(image_name).stem
    for p in candidates:
        if p.exists():
            return p
    # try suffix
    for p in ROOT.rglob(base + '*mask*.png'):
        return p
    return None

# Main
if __name__ == '__main__':
    images = list_input_images()
    methods = ['dod_fade','dod_sam','lol_fade','lol_sam','output']
    per_image = []
    # collect LPIPS pairs
    lpips_pairs = {}
    for img in images:
        ref = INPUT_FOLDER / img
        for m in methods:
            cand = method_image_path(m, img)
            if cand.exists():
                key = f'{m}:{img}'
                lpips_pairs[key] = (str(ref), str(cand))
    # compute LPIPS
    lpips_results = compute_lpips_pairs(lpips_pairs)

    # compute per-method aggregated LPIPS
    agg_lpips = {}
    for m in methods:
        vals = [v for k,v in lpips_results.items() if k.startswith(m+':') and v is not None]
        agg_lpips[m] = float(np.mean(vals)) if len(vals)>0 else None

    # compute FID (method images vs originals)
    agg_fid = {}
    orig_paths = [str(INPUT_FOLDER / img) for img in images]
    for m in methods:
        paths = [str(method_image_path(m, img)) for img in images if method_image_path(m, img).exists()]
        if len(paths)>0:
            fid = compute_fid(orig_paths, paths)
        else:
            fid = None
        agg_fid[m] = fid

    # IoU and Boundary: read from metrics.json if previously computed
    agg_iou = {}
    agg_boundary = {}
    existing_metrics = {}
    if METRICS_OUT.exists():
        try:
            existing_metrics = json.loads(METRICS_OUT.read_text())
        except Exception:
            existing_metrics = {}
    for m in methods:
        agg_iou[m] = existing_metrics.get('iou', {}).get(m, None)
        agg_boundary[m] = existing_metrics.get('boundary_energy', {}).get(m, None)

    def normalize_dict(d):
        vals = [v for v in d.values() if v is not None]
        if len(vals)==0:
            return {k:None for k in d}
        mn = min(vals); mx = max(vals)
        if mx==mn:
            return {k:(0.5 if d[k] is not None else None) for k in d}
        out = {}
        for k,v in d.items():
            out[k] = (v - mn)/(mx - mn) if v is not None else None
        return out
    norm_fid = normalize_dict(agg_fid)
    norm_lpips = normalize_dict(agg_lpips)
    norm_boundary = normalize_dict(agg_boundary)

    composite = {}
    for m in methods:
        if norm_fid.get(m) is None or norm_lpips.get(m) is None:
            composite[m] = None
        else:
            E_norm = norm_boundary.get(m)
            if E_norm is None:
                E_norm = 0.5
                print(f'Warning: no boundary energy for {m}, using placeholder E_norm=0.5')
            composite[m] = 0.5*(1 - E_norm) + 0.3*(1 - norm_fid[m]) + 0.2*(1 - norm_lpips[m])

    # Save metrics (merge into existing to preserve fields from other scripts)
    out = existing_metrics.copy()
    out.pop('boundary', None)
    out.update({
        'fid': agg_fid,
        'lpips': agg_lpips,
        'iou': agg_iou,
        'boundary_energy': agg_boundary,
        'composite': composite,
        'per_image_lpips': lpips_results,
    })
    METRICS_OUT.write_text(json.dumps(out, indent=2))
    print('Wrote metrics to', METRICS_OUT)

    # write simple latex table snippet
    lines = []
    lines.append('\\begin{tabularx}{\\linewidth}{|l|c|c|c|c|c|}')
    lines.append('\\hline')
    lines.append('Method & FID & LPIPS (FG) & IoU & Boundary $E$ & Composite $Q$ \\\\')
    lines.append('\\hline')
    name_map = {'dod_fade':'DOD-FADE','dod_sam':'DOD-SAM','lol_fade':'LOL-FADE','lol_sam':'LOL-SAM','output':'OUTPUT'}
    for m in methods:
        fid_str = f"{agg_fid[m]:.2f}" if agg_fid[m] is not None else '--'
        lp_str = f"{agg_lpips[m]:.4f}" if agg_lpips[m] is not None else '--'
        iou_str = '--'
        be_str = '--'
        comp_str = f"{composite[m]:.4f}" if composite[m] is not None else '--'
        lines.append(f"{name_map.get(m,m)} & {fid_str} & {lp_str} & {iou_str} & {be_str} & {comp_str} \\\\")
    lines.append('\\hline')
    lines.append('\\end{tabularx}')
    LATEX_SNIPPET.write_text('\n'.join(lines))
    print('Wrote LaTeX snippet to', LATEX_SNIPPET)

    # write per-image CSV
    with open(PER_IMAGE_CSV,'w') as f:
        f.write('image,method,lpips\n')
        for k,v in lpips_results.items():
            try:
                method,image = k.split(':',1)
            except:
                method = 'unknown'
                image = k
            f.write(f"{image},{method},{v}\n")
    print('Wrote per-image CSV to', PER_IMAGE_CSV)
    print('Done')
