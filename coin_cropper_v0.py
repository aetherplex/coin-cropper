#!/usr/bin/env python3
"""
PoC: segment coins/bank-notes from an image and write each to disk.

$ python coin_cropper.py IMG_0632.jpg --out_dir ./out
"""

from pathlib import Path
import argparse, cv2, numpy as np, torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# ---------- CLI arguments ----------
parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, help="input image file")
parser.add_argument("--out_dir", type=str, default="out", help="where to save crops")
parser.add_argument("--sam_ckpt", type=str,
                    default="sam_vit_h_4b8939.pth", help="SAM checkpoint")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

# ---------- Load model ----------
print(f"[+] Loading SAM on {args.device} …")
sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt)
sam.to(args.device)
amg = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=1,
)

# ---------- Run segmentation ----------
img_bgr = cv2.imread(args.image)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
raw_masks = amg.generate(img_rgb)

# ---------- Post-filter ----------
H, W = img_rgb.shape[:2]
MIN_AREA  = 8_000       # tweak ↓ until you stop picking up binder rings
MAX_AREA  = 300_000
ROUNDNESS = 0.60        # 1 = perfect circle

good = []
for m in raw_masks:
    area = m["area"]
    if not (MIN_AREA < area < MAX_AREA):
        continue

    # compute roundness via perimeter² / (4π·area)
    mask = m["segmentation"].astype(np.uint8)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:  continue
    perim = cv2.arcLength(contours[0], True)
    roundness = 4 * np.pi * area / (perim**2 + 1e-5)
    if roundness < ROUNDNESS:  # too elongated
        continue

    good.append(m["bbox"])  # (x0,y0,w,h)

# ---------- Crop & save ----------
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
for i,(x,y,w,h) in enumerate(good, start=1):
    # add a little padding
    pad = 6
    x0, y0 = max(x-pad,0), max(y-pad,0)
    x1, y1 = min(x+w+pad, W), min(y+h+pad, H)
    crop = img_bgr[y0:y1, x0:x1]
    cv2.imwrite(str(out_dir / f"crop_{i:04d}.png"), crop)

print(f"[✓] Wrote {len(good)} crops to {out_dir.resolve()}")
