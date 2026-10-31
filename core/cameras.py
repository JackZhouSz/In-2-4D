"""Orthographic multi-view cameras -> perspective cameras for the Gaussian rasterizer.

The six views are orthographic at elevation 0 and azimuths 0/45/90/180/270/315 (front = -Y, world Z-up); the ortho
image spans [-1, 1] world units. Every view is a look-at camera on one sphere. The rasterizer is perspective, so
each view is approximated by a far (distance D) narrow camera whose frustum is 2 world units wide at the object,
with znear = D - 5, zfar = D + 5 (a fixed znear=0.01 / zfar=100 would clip everything at D=200).
"""
import json
import math
from pathlib import Path

import numpy as np
import torch


ERA3D_VIEWS = ["front", "front_right", "right", "back", "left", "front_left"]


ERA3D_AZIMUTHS = [0.0, 45.0, 90.0, 180.0, 270.0, 315.0]
ERA3D_ELEVATION = 0.0


def far_c2w(view: str, D: float) -> np.ndarray:
    """Look-at camera on a sphere of radius D, Z-up, at the view's azimuth."""
    return look_at_c2w(ERA3D_AZIMUTHS[ERA3D_VIEWS.index(view)], ERA3D_ELEVATION, D)


def far_fov(D: float, half_extent: float = 1.0) -> float:
    return 2 * math.atan(half_extent / D)


def look_at_c2w(azimuth_deg: float, elevation_deg: float, radius: float) -> np.ndarray:
    """Blender c2w on a Z-up orbit; azimuth 0 = front camera at -Y (same convention as Era3D's front)."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    eye = radius * np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, -fwd, eye
    return c2w


def write_transforms(path, frames, camera_angle_x):
    """Blender-style transforms_*.json; frames = [{file_path, transform_matrix, time, time_index, view_index}]."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"camera_angle_x": camera_angle_x, "frames": frames}, f, indent=1)



def projection_matrix(znear, zfar, fovx, fovy) -> torch.Tensor:
    """OpenGL-style perspective projection (row-vector convention)."""
    tx, ty = math.tan(fovx / 2), math.tan(fovy / 2)
    P = torch.zeros(4, 4)
    P[0, 0] = 1.0 / tx
    P[1, 1] = 1.0 / ty
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class GSCamera:
    """Attributes consumed by the renderer and by the tracking loss (world_view_transform etc. are
    stored transposed, row-vector convention)."""

    def __init__(self, c2w_blender: np.ndarray, fovx: float, width: int, height: int, time: float,
                 znear: float, zfar: float, device="cuda"):
        c2w = np.array(c2w_blender, dtype=np.float64)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        self.R = w2c[:3, :3].T
        self.T = w2c[:3, 3]
        self.FoVx = fovx
        self.FoVy = 2 * math.atan(math.tan(fovx / 2) * height / width)
        self.image_width, self.image_height = width, height
        self.time = float(time)
        self.znear, self.zfar = znear, zfar
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32).T.contiguous().to(device)
        self.projection_matrix = projection_matrix(znear, zfar, self.FoVx, self.FoVy).T.to(device)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.fx = width / (2 * math.tan(self.FoVx / 2))
        self.fy = height / (2 * math.tan(self.FoVy / 2))

    def at_time(self, time: float) -> "GSCamera":
        cam = object.__new__(GSCamera)
        cam.__dict__.update(self.__dict__)
        cam.time = float(time)
        return cam

    def resized(self, width: int, height: int) -> "GSCamera":
        """Same pose and FoV at another resolution (low-res training augmentation)."""
        cam = self.at_time(self.time)
        cam.image_width, cam.image_height = width, height
        cam.fx = width / (2 * math.tan(self.FoVx / 2))
        cam.fy = height / (2 * math.tan(self.FoVy / 2))
        return cam


def era3d_camera(view: str, D: float, size: int, time: float = 0.0, device="cuda") -> GSCamera:
    return GSCamera(far_c2w(view, D), far_fov(D), size, size, time, znear=D - 5, zfar=D + 5, device=device)


def project_points(xyz: torch.Tensor, cam: GSCamera):
    """World points [N, 3] -> pixel coords [N, 2] and camera-space depth [N]."""
    P = cam.full_proj_transform
    hom = xyz @ P[:3, :] + P[3:4, :]
    w = hom[:, 3]
    ndc = hom[:, :2] / (w[:, None] + 1e-4)
    px = torch.stack([((ndc[:, 0] + 1.0) * cam.image_width - 1.0) * 0.5,
                      ((ndc[:, 1] + 1.0) * cam.image_height - 1.0) * 0.5], -1)
    return px, w


def unproject_points(px: torch.Tensor, depth: torch.Tensor, cam: GSCamera) -> torch.Tensor:
    """Inverse of project_points: pixel coords [N, 2] + camera-space depth [N] -> world points [N, 3]."""
    ndc_x = (2.0 * px[:, 0] + 1.0) / cam.image_width - 1.0
    ndc_y = (2.0 * px[:, 1] + 1.0) / cam.image_height - 1.0
    x = ndc_x * math.tan(cam.FoVx / 2) * depth
    y = ndc_y * math.tan(cam.FoVy / 2) * depth
    hom = torch.stack([x, y, depth, torch.ones_like(depth)], -1)
    return (hom @ cam.world_view_transform.inverse())[:, :3]
