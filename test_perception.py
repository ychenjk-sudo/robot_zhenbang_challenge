"""
逐个测试三种感知方案，告诉你哪个在你的机器/摄像头上真的 work。

用法:
    cd challenge_pkg/robot_zhenbang_challenge
    python test_perception.py          # 用摄像头实时测试，按 q 退出
    python test_perception.py --image /path/to/photo.jpg  # 用图片测试
    python test_perception.py --backend color    # 只测颜色分割
    python test_perception.py --backend yolo     # 只测 YOLO
    python test_perception.py --backend owlvit   # 只测 OWL-ViT (慢)

界面说明:
    - 左上角显示当前 backend 和 FPS
    - 绿框 = 检测到的物体
    - 按 1/2/3 切换目标物体 (萝卜/纸巾/可乐)
    - 按 n 切换下一个 backend
    - 按 s 保存当前帧到 /tmp/test_frame.jpg
    - 按 q 退出
"""
import sys
import time
import argparse
import numpy as np
import cv2

NAMES = ["🥕 萝卜", "🧻 纸巾", "🥤 可乐"]
PROMPTS = [
    "a fresh orange carrot",
    "a roll of white toilet paper",
    "a red can of Coca-Cola",
]
BACKENDS = ["vlm", "color", "yolo", "owlvit"]


def make_detector(backend: str):
    from detector import ObjectDetector
    return ObjectDetector(
        prompts=PROMPTS,
        names=NAMES,
        backend=backend,
    )


def draw_info(frame, detections, backend, fps, target_idx):
    out = frame.copy()
    H, W = out.shape[:2]

    # Draw all detections
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        is_target = (d["name_idx"] == target_idx)
        color = (0, 255, 0) if is_target else (255, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if is_target else 2)
        label = f"{d['name']} {d['score']*100:.0f}%"
        cv2.putText(out, label, (x1 + 2, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Top status bar
    cv2.rectangle(out, (0, 0), (W, 60), (30, 30, 30), -1)
    cv2.putText(out, f"backend: {backend}   FPS: {fps:.1f}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(out, f"target: {NAMES[target_idx]}   detections: {len(detections)}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 1)

    # Bottom help
    cv2.rectangle(out, (0, H - 30), (W, H), (30, 30, 30), -1)
    cv2.putText(out, "1/2/3: target   n: next backend   s: save   q: quit",
                (10, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # No-detection warning
    target_found = any(d["name_idx"] == target_idx for d in detections)
    if not target_found:
        cv2.putText(out, f"NOT FOUND: {NAMES[target_idx]}",
                    (W // 2 - 120, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out


def test_single_image(image_path: str, backend: str):
    """Non-interactive test on a single image, prints results."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: cannot read {image_path}")
        sys.exit(1)

    print(f"\n=== Testing backend={backend} on {image_path} ===")
    det = make_detector(backend)
    t0 = time.time()
    dets = det.detect(frame, top_k_per_class=2)
    elapsed = time.time() - t0
    print(f"Time: {elapsed*1000:.0f} ms   Detections: {len(dets)}")
    for d in dets:
        print(f"  {d['name']}  score={d['score']:.3f}  bbox={d['bbox']}")
    if not dets:
        print("  *** NO DETECTIONS ***")
    out = det.annotate(frame, dets)
    out_path = f"/tmp/test_result_{backend}.jpg"
    cv2.imwrite(out_path, out)
    print(f"Annotated image saved to {out_path}")


def run_live(cam_index: int, initial_backend: str):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {cam_index}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    backend_idx = BACKENDS.index(initial_backend) if initial_backend in BACKENDS else 0
    target_idx = 0
    detector = None
    fps = 0.0
    t_prev = time.time()

    print("\nCamera open. Loading first backend...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        # Lazy-load detector when backend changes
        if detector is None:
            backend = BACKENDS[backend_idx]
            print(f"\nLoading backend: {backend} ...")
            try:
                detector = make_detector(backend)
                # Force load now so first frame isn't slow
                detector._ensure_loaded()
                print(f"  OK: {backend} ready (active={detector._active_backend})")
            except Exception as e:
                print(f"  FAILED: {e}")
                detector = None
                backend_idx = (backend_idx + 1) % len(BACKENDS)
                continue

        backend = BACKENDS[backend_idx]
        try:
            dets = detector.detect(frame, top_k_per_class=2)
        except Exception as e:
            dets = []
            print(f"detect() error: {e}")

        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        disp = draw_info(frame, dets, backend, fps, target_idx)
        cv2.imshow("Perception Test", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            target_idx = 0
        elif key == ord('2'):
            target_idx = 1
        elif key == ord('3'):
            target_idx = 2
        elif key == ord('n'):
            backend_idx = (backend_idx + 1) % len(BACKENDS)
            detector = None
            print(f"\nSwitching to: {BACKENDS[backend_idx]}")
        elif key == ord('s'):
            path = "/tmp/test_frame.jpg"
            cv2.imwrite(path, frame)
            print(f"Saved frame to {path}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="静态图片路径，不用摄像头")
    parser.add_argument("--backend", default="vlm",
                        choices=["vlm", "color", "yolo", "owlvit", "all"],
                        help="要测试的 backend (all = 依次测全部)")
    parser.add_argument("--cam", type=int, default=0, help="摄像头编号")
    args = parser.parse_args()

    if args.image:
        targets = BACKENDS if args.backend == "all" else [args.backend]
        for b in targets:
            test_single_image(args.image, b)
    else:
        run_live(args.cam, args.backend)


if __name__ == "__main__":
    main()
