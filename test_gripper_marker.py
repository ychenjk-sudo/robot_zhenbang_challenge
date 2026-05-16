"""
一次性脚本: 测试 wrist cam 能不能看到夹爪左指尖的黑色标记.

跑这个时机:
  1. 先停掉 main.py (它在占摄像头)
  2. 把夹爪摆成你常用的 observe / hover 姿态 (machine 通电就行, 不需要跑游戏)
  3. python test_gripper_marker.py
  4. 看终端输出 + 看保存的图片 marker_test.png / marker_test_annotated.png

会做 3 件事:
  A. 抓一帧原始画面 → 保存为 marker_test.png
  B. 检测画面里所有的"暗块" → 标注到 marker_test_annotated.png
  C. 报告: 找到几个候选, 哪个最可能是夹爪标记 (画面下部 + 面积合理 的)
"""
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# 配置 — 调这几个值来 fine-tune 检测
# ---------------------------------------------------------------------------

# 黑色阈值: HSV 的 V (亮度) < V_MAX 视为黑
V_MAX = 60          # 0-255. 太低会漏检, 太高会把阴影/暗色物体误检
S_MAX = 80          # 0-255. 黑色饱和度低. 排除深色但有色物体 (如深红萝卜)

# 最小 / 最大面积 (像素²). 排除噪点和过大的物体
MIN_AREA = 200
MAX_AREA = 30000

# "可能是夹爪标记" 的位置筛选 — 标记应该出现在画面下半部 (因为夹爪在 wrist cam 下方)
# 这里给一个软分数, 越靠下越像
def gripper_likelihood(cy_norm: float) -> float:
    """cy_norm ∈ [0,1] = 中心点 y 占画面高度的比例. 1 = 最下."""
    # 在 [0.4, 1.0] 区间线性增长, 上半部分得分=0
    if cy_norm < 0.4:
        return 0.0
    return (cy_norm - 0.4) / 0.6   # 0.4→0, 1.0→1


# ---------------------------------------------------------------------------
def detect_dark_blobs(frame_bgr: np.ndarray):
    """返回所有暗块的 list: [{bbox, area, centroid, score}, ...]"""
    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # 黑色 mask: 低饱和度 + 低亮度
    mask = ((hsv[:, :, 1] < S_MAX) & (hsv[:, :, 2] < V_MAX)).astype(np.uint8) * 255

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = int(cv2.contourArea(c))
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        cy_norm = cy / H
        score = gripper_likelihood(cy_norm)
        blobs.append({
            "bbox": (x, y, x + w, y + h),
            "area": area,
            "centroid": (cx, cy),
            "cy_norm": cy_norm,
            "score": score,
        })
    blobs.sort(key=lambda b: -b["score"])
    return mask, blobs


def main():
    # 默认从 config.yaml 读相机 index
    import yaml
    cfg_path = Path(__file__).with_name("config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cam_index = cfg["camera"]["index"]
    cam_w = cfg["camera"]["width"]
    cam_h = cfg["camera"]["height"]

    print(f"打开摄像头 {cam_index} ({cam_w}x{cam_h})...")
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    if not cap.isOpened():
        print("❌ 打不开摄像头. 是不是 main.py 还在跑? 先 Ctrl+C 退掉它.")
        return

    # 预热 — 前几帧通常是黑屏
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("❌ 读不到画面")
        return

    H, W = frame.shape[:2]
    print(f"✓ 拿到一帧 {W}x{H}")

    # 保存原图
    cv2.imwrite("marker_test_raw.png", frame)
    print("✓ 原图保存到 marker_test_raw.png")

    # 检测
    mask, blobs = detect_dark_blobs(frame)
    cv2.imwrite("marker_test_mask.png", mask)
    print(f"✓ 黑色 mask 保存到 marker_test_mask.png")
    print(f"✓ 找到 {len(blobs)} 个暗块候选 (面积 {MIN_AREA}~{MAX_AREA} 像素²)")

    # 标注
    annotated = frame.copy()
    for i, b in enumerate(blobs):
        x1, y1, x2, y2 = b["bbox"]
        is_top = (i == 0 and b["score"] > 0)
        color = (0, 255, 0) if is_top else (0, 200, 200)
        thick = 3 if is_top else 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
        label = f"#{i+1} area={b['area']} y={b['cy_norm']:.2f} score={b['score']:.2f}"
        cv2.putText(annotated, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        # 标心
        cx, cy = b["centroid"]
        cv2.circle(annotated, (cx, cy), 4, color, -1)

    # 把"夹爪应该出现的下半区"画出来
    cv2.line(annotated, (0, int(H * 0.4)), (W, int(H * 0.4)), (128, 128, 128), 1)
    cv2.putText(annotated, "below = gripper zone", (10, int(H * 0.4) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

    cv2.imwrite("marker_test_annotated.png", annotated)
    print(f"✓ 标注图保存到 marker_test_annotated.png")

    # 报告
    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)
    if not blobs:
        print("❌ 没找到任何暗块")
        print("   可能原因:")
        print("   1. 标记不够黑 — 看 marker_test_raw.png, 是否在画面里能看到?")
        print("   2. 标记被夹爪本体遮住了 — 换个角度贴, 或换更显眼的颜色 (橙/绿/红)")
        print("   3. 阈值太严 — 调本脚本顶部 V_MAX=60 → 100 再试")
    else:
        for i, b in enumerate(blobs[:5]):
            tag = "⭐ 最可能是标记" if i == 0 and b["score"] > 0 else "  "
            print(f"  {tag} #{i+1}: bbox={b['bbox']}, 面积={b['area']}px², "
                  f"中心=({b['centroid'][0]},{b['centroid'][1]}), "
                  f"垂直位置={b['cy_norm']:.2%}, 夹爪相似度={b['score']:.2f}")

        top = blobs[0]
        if top["score"] > 0.3:
            print(f"\n✅ 看起来检测到了!")
            print(f"   标记在画面 ({top['centroid'][0]}, {top['centroid'][1]}) 位置")
            print(f"   面积 {top['area']} 像素²")
            print(f"   后续可以用这个坐标做 gripper-relative 视觉伺服")
        else:
            print(f"\n⚠️ 找到了一些暗块, 但都在画面上半部, 不像夹爪")
            print(f"   可能夹爪标记没在视野里")
            print(f"   把机械臂移到 hover/hit 姿态再跑一次试试")


if __name__ == "__main__":
    main()
