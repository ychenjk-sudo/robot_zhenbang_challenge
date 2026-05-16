"""
WorkspaceMap: 把摄像头像素 + 桌面世界坐标映射到机械臂关节角。

数据来源: calibrate.py 生成的 calibration.npz
    - homography (3x3)        : pixel (u,v) → world (x,y) ∈ [0,1]²
    - grid_xy (N, 2)          : 标定时使用的世界坐标采样点
    - hover_joints (N, 6)     : 每个采样点的 hover 关节角
    - hit_joints (N, 6)       : 每个采样点的 hit 关节角
    - joint_names (6,)        : 关节名称顺序

插值方式: RBF (radial basis function) — 在采样点之间做平滑过渡。
对每个关节独立插值，scipy.interpolate.Rbf。
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class WorkspaceMap:
    def __init__(self, calibration_path: str = "calibration.npz"):
        path = Path(calibration_path)
        if not path.exists():
            raise FileNotFoundError(
                f"标定文件不存在: {calibration_path}。先运行: python calibrate.py")
        data = np.load(path, allow_pickle=True)
        self.path = path
        self.homography: np.ndarray = data["homography"]
        self.grid_xy: np.ndarray   = data["grid_xy"]           # (N, 2)
        self.hover: np.ndarray     = data["hover_joints"]      # (N, 6)
        self.hit:   np.ndarray     = data["hit_joints"]        # (N, 6)
        names = data["joint_names"]
        self.joint_names = [str(n) for n in names.tolist()] if names.dtype.kind in "OSU" else list(names)

        if not (self.grid_xy.shape[0] == self.hover.shape[0] == self.hit.shape[0]):
            raise ValueError("calibration arrays have inconsistent lengths")

        # Build per-joint RBF interpolators (lazy import scipy)
        from scipy.interpolate import Rbf
        x = self.grid_xy[:, 0]
        y = self.grid_xy[:, 1]
        # multiquadric is smooth and well-behaved for small N
        self._hover_rbfs = [Rbf(x, y, self.hover[:, j], function="multiquadric")
                            for j in range(self.hover.shape[1])]
        self._hit_rbfs   = [Rbf(x, y, self.hit[:, j],   function="multiquadric")
                            for j in range(self.hit.shape[1])]
        logger.info("WorkspaceMap loaded: %d points, %d joints from %s",
                    len(self.grid_xy), len(self.joint_names), path)

    # ------------------------------------------------------------------
    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """Apply homography: pixel (u,v) → world (x,y)."""
        p = np.array([u, v, 1.0])
        w = self.homography @ p
        if abs(w[2]) < 1e-9:
            raise ValueError("homography projects to point at infinity")
        return float(w[0] / w[2]), float(w[1] / w[2])

    # ------------------------------------------------------------------
    def joints_at(self, x: float, y: float, mode: str = "hover") -> Dict[str, float]:
        """Return interpolated joint dict at world (x, y)."""
        rbfs = self._hover_rbfs if mode == "hover" else self._hit_rbfs
        vals = [float(r(x, y)) for r in rbfs]
        return dict(zip(self.joint_names, vals))

    # ------------------------------------------------------------------
    def in_bounds(self, x: float, y: float, margin: float = 0.05) -> bool:
        """Is (x, y) inside the calibrated rectangle (with margin)?"""
        return -margin <= x <= 1 + margin and -margin <= y <= 1 + margin

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"WorkspaceMap(N={len(self.grid_xy)}, "
                f"joints={self.joint_names}, src={self.path})")
