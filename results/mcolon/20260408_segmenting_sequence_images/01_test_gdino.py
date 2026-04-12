#!/usr/bin/env python3
"""
Test finetuned GroundingDINO on sequence images.
Outputs annotated PNGs and a CSV of raw detections.
"""

import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

MORPHSEQ_ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = MORPHSEQ_ROOT / "segmentation_sandbox"

GDINO_MODULE = SANDBOX_ROOT / "models" / "GroundingDINO"
sys.path.insert(0, str(GDINO_MODULE))
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "scripts" / "detection_segmentation"))

from grounded_dino_utils import load_groundingdino_model  # noqa: E402
from groundingdino.util.inference import load_image, predict  # noqa: E402

# Paths
GDINO_CONFIG = SANDBOX_ROOT / "configs" / "pipeline_config.yaml"
WEIGHTS_PATH = (
    "/net/trapnell/vol1/home/mdcolon/proj/image_segmentation/"
    "Open-GroundingDino/finetune_output/"
    "finetune_output_run_nick_masks_20250308/checkpoint_best_regular.pth"
)
IMAGE_DIR = Path(__file__).parent / "test_sequence_image"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = {
    "models": {
        "groundingdino": {
            "config": "models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "weights": WEIGHTS_PATH,
        }
    }
}

TEXT_PROMPT = "individual embryo"
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.10

THRESHOLD_SWEEPS = [
    (0.15, 0.10),
]


def draw_detections(image_source: np.ndarray, boxes_xyxy: np.ndarray, logits: np.ndarray, phrases: list, title: str) -> plt.Figure:
    """Draw bounding boxes on image and return figure."""
    h, w = image_source.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(image_source)
    ax.set_title(f"{title}  ({len(boxes_xyxy)} detections)", fontsize=12)
    ax.axis("off")

    for box, conf, phrase in zip(boxes_xyxy, logits, phrases):
        x1, y1, x2, y2 = box * np.array([w, h, w, h])
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 4,
            f"{phrase} {conf:.2f}",
            color="lime", fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"),
        )
    return fig


DEVICE = "cpu"


def run_inference_cpu(model, image_path, text_prompt, box_threshold, text_threshold):
    """Run inference on CPU, returning (boxes_xyxy_norm, logits, phrases, image_source)."""
    import torch
    image_source, image_tensor = load_image(str(image_path))
    boxes_tensor, logits_tensor, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=DEVICE,
    )
    boxes_xywh = boxes_tensor.cpu().numpy() if isinstance(boxes_tensor, torch.Tensor) else np.array(boxes_tensor)
    logits = logits_tensor.cpu().numpy() if isinstance(logits_tensor, torch.Tensor) else np.array(logits_tensor)
    # cxcywh → xyxy
    if len(boxes_xywh) > 0:
        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        boxes_xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    else:
        boxes_xyxy = np.zeros((0, 4))
    return boxes_xyxy, logits, phrases, image_source


def main():
    print("Loading GroundingDINO model (CPU)...")
    model = load_groundingdino_model(CONFIG, device="cpu")
    print("Model loaded.\n")

    images = sorted(IMAGE_DIR.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found in {IMAGE_DIR}")

    all_rows = []

    for image_path in images:
        print(f"--- {image_path.name} ---")
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        best_boxes, best_logits, best_phrases = None, None, None
        best_thresh = None

        for box_thr, text_thr in THRESHOLD_SWEEPS:
            boxes, logits, phrases, image_source = run_inference_cpu(
                model, image_path, TEXT_PROMPT, box_threshold=box_thr, text_threshold=text_thr
            )
            n = len(boxes)
            print(f"  box_thr={box_thr:.2f} text_thr={text_thr:.2f} → {n} detection(s)")
            if best_boxes is None:
                best_boxes, best_logits, best_phrases = boxes, logits, phrases
                best_thresh = (box_thr, text_thr)
            if n > 0:
                best_boxes, best_logits, best_phrases = boxes, logits, phrases
                best_thresh = (box_thr, text_thr)
                break

        boxes_xyxy = best_boxes  # already in xyxy normalized from run_inference_cpu
        print(f"  Using threshold box={best_thresh[0]:.2f} text={best_thresh[1]:.2f}")
        for i, (box, conf, phrase) in enumerate(zip(boxes_xyxy, best_logits, best_phrases)):
            print(f"    [{i}] {phrase} conf={conf:.3f}  xyxy={box.round(3).tolist()}")
            all_rows.append({
                "image": image_path.name,
                "box_x1": round(float(box[0]), 4),
                "box_y1": round(float(box[1]), 4),
                "box_x2": round(float(box[2]), 4),
                "box_y2": round(float(box[3]), 4),
                "confidence": round(float(conf), 4),
                "phrase": phrase,
                "box_threshold": best_thresh[0],
                "text_threshold": best_thresh[1],
            })

        # Save annotated image
        fig = draw_detections(img_np, boxes_xyxy, best_logits, best_phrases, image_path.stem)
        out_path = OUTPUT_DIR / f"{image_path.stem}_detected.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}\n")

    # Save CSV
    csv_path = OUTPUT_DIR / "detections.csv"
    fieldnames = ["image", "box_x1", "box_y1", "box_x2", "box_y2", "confidence", "phrase", "box_threshold", "text_threshold"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Detections saved to: {csv_path}")
    print(f"Total detections across all images: {len(all_rows)}")


if __name__ == "__main__":
    main()
