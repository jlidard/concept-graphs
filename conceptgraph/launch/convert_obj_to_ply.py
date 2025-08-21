from __future__ import annotations


import numpy as np
import open3d as o3d



def _normalize_rgb(vals: np.ndarray) -> np.ndarray:
    """
    Ensure colors are in [0,1]. If any channel > 1.0, assume 0-255 and scale.
    """
    if np.any(vals > 1.0):
        vals = vals / 255.0
    return np.clip(vals, 0.0, 1.0)


def obj_to_ply_with_colors(obj_path: str, ply_path: str) -> None:
    """
    Convert a vertex-only OBJ to PLY, preserving per-vertex colors when present.

    Supports common non-standard patterns like:
      v x y z r g b         (RGB as 0-1 or 0-255)
      v x y z r g b a       (ignores alpha)
    If only positions are present, writes an uncolored PLY.
    """
    points = []
    colors = []

    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            # parts[0] == 'v'
            nums = [float(x) for x in parts[1:]]
            if len(nums) < 3:
                continue  # malformed
            x, y, z = nums[0], nums[1], nums[2]
            points.append([x, y, z])

            # Heuristics for color fields (OBJ vertex color is non-standard)
            # If 3+ extra values, treat next three as RGB; ignore any alpha / extras.
            if len(nums) >= 6:
                r, g, b = nums[3], nums[4], nums[5]
                colors.append([r, g, b])

    if not points:
        raise ValueError(f"No vertices found in {obj_path}")

    pts_np = np.asarray(points, dtype=float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)

    if len(colors) == len(points):
        cols_np = _normalize_rgb(np.asarray(colors, dtype=float))
        pcd.colors = o3d.utility.Vector3dVector(cols_np)

    # Write colored PLY (binary by default). Meshlab reads PLY vertex colors.
    ok = o3d.io.write_point_cloud(ply_path, pcd)  # write_ascii/compressed available if needed
    if not ok:
        raise IOError(f"Failed to write {ply_path}")


if __name__ == "__main__":
    obj_path = "/home/pbrick/dev/data/250814_ECL_200_2_0.01_20_1/ptcloud/ptcloud.obj"
    ply_path = "/home/pbrick/dev/data/250814_ECL_200_2_0.01_20_1/ptcloud/ptcloud.ply"

    obj_to_ply_with_colors(obj_path, ply_path)
