#!/usr/bin/env python3
"""
PoC: Zero-shot coin cropping via Grounding DINO → SAM

Usage:
  # first, download the checkpoints/configs:
  #   wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
  #   wget https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
  #   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

  python grounded_coin_cropper.py \
    --image test_01.jpg \
    --gd_config GroundingDINO_SwinT_OGC.py \
    --gd_checkpoint groundingdino_swint_ogc.pth \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --out_dir crops \
    --device cuda
"""

import argparse, os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert

# Grounding DINO inference (zero-shot detector) :contentReference[oaicite:0]{index=0}
from groundingdino.util.inference import load_model as load_gd_model, \
                                         load_image as load_gd_image, \
                                         predict as gd_predict

# Segment-Anything :contentReference[oaicite:1]{index=1}
from segment_anything import sam_model_registry, SamPredictor

def parse_args():
    p = argparse.ArgumentParser(description="Grounded coin cropper")
    p.add_argument("--image",           type=str, required=True,
                   help="Path to input image")
    p.add_argument("--gd_config",       type=str, required=True,
                   help="Grounding DINO config .py file")
    p.add_argument("--gd_checkpoint",   type=str, required=True,
                   help="Grounding DINO weights .pth file")
    p.add_argument("--sam_checkpoint",  type=str, required=True,
                   help="SAM checkpoint .pth file")
    p.add_argument("--out_dir",         type=str, default="crops",
                   help="Where to save coin crops")
    p.add_argument("--box_threshold",   type=float, default=0.3,
                   help="DINO box confidence threshold")
    p.add_argument("--text_threshold",  type=float, default=0.25,
                   help="DINO text-match threshold")
    p.add_argument("--device",          type=str, default="cuda",
                   help="torch device")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1) Load Grounding DINO ──
    print(f"[+] Loading Grounding DINO on {args.device} …")
    gd_model = load_gd_model(
        args.gd_config,
        args.gd_checkpoint,
        device=args.device
    )

    # ── 2) Run zero-shot detection for “coin” ──
    print("[+] Running DINO zero-shot detection for 'coin' …")
    image_source, image_tensor = load_gd_image(args.image)
    boxes, scores, phrases = gd_predict(
        model=gd_model,
        image=image_tensor,
        caption="coin",
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device
    )

    # ── 3) Filter & convert boxes to pixel xyxy ──
    H, W = image_source.shape[:2]
    # keep only highest-confidence boxes (optional)
    keep = scores > args.box_threshold
    boxes = boxes[keep]

    # convert from cxcywh (0–1) to xyxy pixel coords
    xyxy = box_convert(boxes, "cxcywh", "xyxy")
    xyxy = (xyxy * torch.tensor([W, H, W, H], device=xyxy.device))\
           .int().cpu().numpy().tolist()

    # ── 4) Load SAM & predictor ──
    print(f"[+] Loading SAM vit_h on {args.device} …")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
    sam.to(args.device)
    predictor = SamPredictor(sam)
    # SAM expects RGB
    img_bgr  = cv2.imread(args.image)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    # ── 5) Mask & crop each box ──
    print(f"[+] Segmenting {len(xyxy)} DINO boxes with SAM …")
    for idx, box in enumerate(xyxy, start=1):
        masks, scores, logits = predictor.predict(
            box=np.array(box),
            multimask_output=False
        )
        mask = masks[0].astype(bool)

        # tightest bounding from mask
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        crop = img_bgr[y0:y1, x0:x1]
        out_path = Path(args.out_dir) / f"crop_{idx:03d}.png"
        cv2.imwrite(str(out_path), crop)

    print(f"[✓] Done! Saved {len(xyxy)} crops to '{args.out_dir}/'")

if __name__ == "__main__":
    main()
