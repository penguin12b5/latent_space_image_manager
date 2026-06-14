# Stylization Ablation Study

## Purpose

Generate quantitative data and visual evidence to support the paper's 3rd contribution claim: that extrapolating mask alpha beyond [0,1] produces stylization effects, validated by ablations and controls (mask-scale sweeps, latent histograms, pixel-space alpha>1 controls, and comparisons to standard filters and a diffusion-based baseline).

## Script

`generate_stylization_ablation.py`

### Usage

```bash
cd /workarea/latent_space_image_manager
SAM_CHECKPOINT=models/sam_vit_b_01ec64.pth python generate_stylization_ablation.py
```

### What it generates

For each of the 11 benchmark images (car1-3, cat1-3, dog1-2, horse1-3):

| Ablation | Output files | Data captured |
|---|---|---|
| **Mask-scale sweep** | `{stem}_alpha_{0.0,0.5,1.0,1.5,2.0}.png` | Per-alpha composite images + latent stats (mean, std, skewness, min, max, fraction out-of-range) |
| **Pixel-space alpha>1 control** | `{stem}_pixel_alpha_{1.5,2.0}.png` | Same metrics but via RGB scaling (no latent-space involvement) |
| **Filter baselines** | `{stem}_{unsharp,bilateral,clahe}.png` | Unsharp mask, bilateral filter, CLAHE comparisons |

Total: 110 output images (11 images x 10 variants).

## Output locations

- **Images**: `images/output/ablation/`
- **Full per-image results**: `results/stylization_ablation_results.json`
- **Aggregated CSV**: `results/stylization_ablation_summary.csv`
- **Summary table**: printed to stdout

## Metrics per variant

| Metric | Description |
|---|---|
| Edge blending score | Gradient magnitude at boundary band; higher = smoother seam |
| Background smoothness | Laplacian energy in background region; higher = smoother |
| Object preservation | SSIM of foreground object vs. original |
| Composite quality score | Weighted aggregate: 0.5E + 0.3B + 0.2O |
| Stylization strength | Ratio of foreground Laplacian energy (result / original) |

For latent-space methods only, additional latent activation statistics are recorded before decoding:

| Statistic | Description |
|---|---|
| mean | Mean of all latent activations in z_merged |
| std | Standard deviation |
| skewness | Third moment / std^3 |
| min / max | Extremes of the latent distribution |
| frac_outside_1 | Fraction of activations with |z| > 1 |
| frac_outside_3 | Fraction of activations with |z| > 3 |

## Ablation details

### 1. Mask-scale sweep (`run_lol_sam_with_alpha`)

Mirrors the LOL-SAM pipeline from `processimage.py`. For each alpha in {0.0, 0.5, 1.0, 1.5, 2.0}, the SAM mask is normalized and then scaled:

```
mask = mask / 255.0 * alpha
blended = background * (1 - mask) + foreground * mask
```

Latent activation statistics are captured on `z_merged` immediately before `decode()`. Expected outcome: monotonic increase in std and out-of-range fraction as alpha increases, supporting the claim that decoder nonlinearities (clipping/saturation) drive the stylized appearance.

### 2. Pixel-space alpha>1 control (`run_pixel_alpha_control`)

Same detection and SAM masking, but operates entirely in pixel space: foreground RGB values are multiplied by alpha and clamped to [0, 255], then composited onto the VAE-decoded background. This control isolates whether the stylization effect requires latent-space operations or can be reproduced by simple pixel scaling.

### 3. Filter baselines (`run_filter_baselines`)

Applied to the original image without any latent-space processing:

- **Unsharp mask**: `cv2.addWeighted(image, 1.5, GaussianBlur(image), -0.5, 0)`
- **Bilateral filter**: `cv2.bilateralFilter(d=9, sigmaColor=75, sigmaSpace=75)`
- **CLAHE**: Contrast-limited adaptive histogram equalization on L channel (clipLimit=2.0, tileGridSize=8x8)

These test whether the stylization appearance can be reproduced by standard image processing filters.

## Dependencies

All from `requirements.txt`: torch, torchvision, diffusers, segment-anything, opencv-python-headless, scikit-image, numpy, matplotlib, PIL.

## Performance notes

- SAM inference is the bottleneck: ~5-6 minutes per image on CPU (ViT-B), ~30 seconds on GPU.
- Full 11-image run: ~50-60 minutes CPU, ~5-10 minutes GPU.
- The script reloads SAM per subject per alpha value. For faster runs, a batched SAM predictor could be added as a future optimization.
