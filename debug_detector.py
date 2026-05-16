"""
OWL-ViT diagnostics: prints RAW scores before threshold filtering.
Run this standalone to see exactly what OWL-ViT is outputting.

Usage:
    cd challenge_pkg/robot_zhenbang_challenge
    python debug_detector.py [--image path/to/test.jpg]

If no --image given, tries to grab one frame from camera index 0.
"""
import sys
import argparse
import numpy as np


PROMPTS = [
    "a fresh orange carrot",
    "a roll of white toilet paper",
    "a red can of Coca-Cola",
]
NAMES = ["萝卜", "纸巾", "可乐"]


def grab_frame(cam_index: int = 0):
    import cv2
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def run_owlvit(frame_bgr, device: str):
    import torch
    import cv2
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    print(f"\n[OWL-ViT] Loading on device={device} ...")
    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    model.eval()
    print(f"[OWL-ViT] Model ready.")

    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(text=[PROMPTS], images=rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # ------ RAW logits before threshold ------
    raw_scores = outputs.logits_per_image  # shape [1, num_boxes, num_queries]
    print(f"\n[RAW logits] shape={raw_scores.shape}")
    sig = torch.sigmoid(raw_scores[0])  # [num_boxes, num_queries]
    for qi, name in enumerate(NAMES):
        col = sig[:, qi]
        top5_vals, top5_idx = col.topk(min(5, col.shape[0]))
        print(f"  {name}: top-5 scores = {[f'{v:.4f}' for v in top5_vals.tolist()]}")

    # ------ post_process at very low threshold ------
    target_sizes = torch.tensor([[H, W]], device=device)
    for thresh in [0.001, 0.005, 0.01, 0.05, 0.10]:
        results = proc.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=thresh,
        )[0]
        n = len(results["scores"])
        print(f"  threshold={thresh:.3f} → {n} detections")
        if n and thresh <= 0.01:
            for s, l, b in zip(
                results["scores"].tolist(),
                results["labels"].tolist(),
                results["boxes"].tolist(),
            ):
                print(f"    {NAMES[l]} score={s:.4f} bbox={[int(x) for x in b]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to test image (jpg/png)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index if no image")
    parser.add_argument("--device", default=None, help="cpu / mps / cuda")
    args = parser.parse_args()

    import cv2
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: cannot read {args.image}", file=sys.stderr)
            sys.exit(1)
        print(f"[Image] loaded {args.image}, shape={frame.shape}")
    else:
        print(f"[Camera] grabbing frame from index {args.cam} ...")
        frame = grab_frame(args.cam)
        if frame is None:
            print("ERROR: no camera frame. Pass --image path/to/photo.jpg", file=sys.stderr)
            sys.exit(1)
        cv2.imwrite("/tmp/debug_frame.jpg", frame)
        print(f"[Camera] frame saved to /tmp/debug_frame.jpg, shape={frame.shape}")

    import torch
    if args.device:
        device = args.device
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\n=== Test 1: device={device} ===")
    run_owlvit(frame, device)

    if device == "mps":
        print(f"\n=== Test 2: device=cpu (compare MPS vs CPU scores) ===")
        run_owlvit(frame, "cpu")


if __name__ == "__main__":
    main()
