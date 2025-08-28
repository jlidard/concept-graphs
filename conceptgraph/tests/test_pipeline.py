
"""
cfslam_pipeline.py
-------------------
A single Python module that replaces the shell entrypoints and exposes
typed functions to run each pipeline component directly (no .sh calls).

Requirements:
  - Python 3.11+
  - Uses pathlib instead of os.path
  - Respects environment variables ZED_ROOT and ZED_CONFIG_PATH used by the original scripts

Provided functions:
  - extract_2d_classes(scene_name, *, class_set="ram", box_threshold=0.2, text_threshold=0.2, stride=5, add_bg_classes=True, accumu_classes=True, exp_suffix="withbg_allclasses")
  - run_map_classes(scene_name, *, threshold=1.0)
  - visualize_classes(scene_name, *, downsample_ratio=0.01, pkl_filename="full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz")
  - run_scene_graph_new(scene_name, *, pkl_filename="full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz", class_set="ram")
  - run_generate_vertices(scene_name, *, pkl_filename="full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz", downsample_ratio=0.01, save_partial_pcd_filename="partial_pcd.ply", full_pc_relpath="ptcloud/ptcloud.ply")
  - run_all(scene_name): executes the default end-to-end sequence analogous to run_all.sh

Notes:
  - This module does *not* execute anything on import.
  - The functions wrap the Python scripts referenced by the shell files you provided.
  - If ZED_ROOT or ZED_CONFIG_PATH are not set, a clear error is raised.
"""

from __future__ import annotations

from pathlib import Path
from subprocess import run as _run, CalledProcessError
import os
from typing import Sequence


def _require_env(var: str) -> str:
    """Return the value of an environment variable or raise a helpful error."""
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(f"Environment variable {var} is required but not set.")
    return val


def _to_str_flag(flag: str, value: bool) -> list[str]:
    """Return [flag] if value is True, else []."""
    return [flag] if value else []


def _exec(cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    """Execute a command with check=True and nice printing."""
    print("\n>>>", " ".join(cmd))
    try:
        _run(list(cmd), check=True, cwd=str(cwd) if cwd else None)
    except CalledProcessError as e:
        raise RuntimeError(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}") from e


# --- Component functions ---

def extract_2d_classes(
    scene_name: str,
    *,
    class_set: str = "ram",
    box_threshold: float = 0.2,
    text_threshold: float = 0.2,
    stride: int = 5,
    add_bg_classes: bool = True,
    accumu_classes: bool = True,
    exp_suffix: str = "withbg_allclasses",
    scripts_dir: Path | None = None,
) -> None:
    """Run scripts/generate_gsa_results.py for a scene."""
    zed_root = _require_env("ZED_ROOT")
    zed_config = _require_env("ZED_CONFIG_PATH")
    scripts_dir = scripts_dir or Path("conceptgraph/scripts")
    cmd: list[str] = [
        "python", str(scripts_dir / "generate_gsa_results.py"),
        "--dataset_root", zed_root,
        "--dataset_config", zed_config,
        "--scene_id", scene_name,
        "--class_set", class_set,
        "--box_threshold", str(box_threshold),
        "--text_threshold", str(text_threshold),
        "--stride", str(stride),
        *(_to_str_flag("--add_bg_classes", add_bg_classes)),
        *(_to_str_flag("--accumu_classes", accumu_classes)),
        "--exp_suffix", exp_suffix,
    ]
    _exec(cmd)


def run_map_classes(
    scene_name: str,
    *,
    threshold: float = 1.0,
    slam_dir: Path | None = None,
) -> None:
    """Run slam/cfslam_pipeline_batch.py for a scene (ConceptGraphs-Detect step)."""
    zed_root = _require_env("ZED_ROOT")
    zed_config = _require_env("ZED_CONFIG_PATH")
    slam_dir = slam_dir or Path("conceptgraph/slam")
    cmd: list[str] = [
        "python", str(slam_dir / "cfslam_pipeline_batch.py"),
        f"dataset_root={zed_root}",
        f"dataset_config={zed_config}",
        "stride=5",
        f"scene_id={scene_name}",
        "spatial_sim_type=overlap",
        "mask_conf_threshold=0.2",
        "match_method=sim_sum",
        f"sim_threshold={threshold}",
        "dbscan_eps=0.05",
        "dbscan_min_points=5",
        "gsa_variant=ram_withbg_allclasses",
        "skip_bg=False",
        "max_bbox_area_ratio=10.0",
        "save_suffix=OUTSIDE",
        "obj_min_detections=1",
        "dbscan_remove_noise=False",
        "downsample_voxel_size=0.01",
        "mask_area_threshold=5",
        "min_points_threshold=16",
        "vis_render=False",
        "debug_render=False",
        "save_objects_all_frames=True",
    ]
    _exec(cmd)


def visualize_classes(
    scene_name: str,
    *,
    downsample_ratio: float = 0.01,
    pcd_dirname: str = "pcd_saves",
    pkl_filename: str = "full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz",
    scripts_dir: Path | None = None,
) -> None:
    """Run scripts/visualize_cfslam_results.py for a scene."""
    zed_root = Path(_require_env("ZED_ROOT"))
    scripts_dir = scripts_dir or Path("conceptgraph/scripts")
    result_path = zed_root / scene_name / pcd_dirname / pkl_filename
    cmd: list[str] = [
        "python", str(scripts_dir / "visualize_cfslam_results.py"),
        "--result_path", str(result_path),
        "--downsample_ratio", str(downsample_ratio),
    ]
    _exec(cmd)


def run_scene_graph_new(
    scene_name: str,
    *,
    pkl_filename: str = "full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz",
    class_set: str = "ram",
    sg_cache_dirname: str = "sg_cache",
    scenegraph_dir: Path | None = None,
    scripts_dir: Path | None = None,
) -> None:
    """Run scenegraph/build_scenegraph_cfslam_gpt.py and scripts/extract_node_captions.py."""
    zed_root = Path(_require_env("ZED_ROOT"))
    scenegraph_dir = scenegraph_dir or Path("conceptgraph/scenegraph")
    scripts_dir = scripts_dir or Path("conceptgraph/scripts")
    cachedir = zed_root / scene_name / sg_cache_dirname
    mapfile = zed_root / scene_name / "pcd_saves" / pkl_filename

    # Step 1: preprocess images
    cmd1: list[str] = [
        "python", str(scenegraph_dir / "build_scenegraph_cfslam_gpt.py"),
        "--mode", "preprocess_images",
        "--masking_option", "red_outline",
        "--cachedir", str(cachedir),
        "--mapfile", str(mapfile),
    ]
    _exec(cmd1)

    # Step 2: extract captions
    cmd2: list[str] = [
        "python", str(scripts_dir / "extract_node_captions.py"),
        "--root_dir", str(cachedir),
        "--image_dir", "cfslam_captions_gpt_debug",
    ]
    _exec(cmd2)


def run_generate_vertices(
    scene_name: str,
    *,
    pkl_filename: str = "full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz",
    downsample_ratio: float = 0.01,
    save_partial_pcd_filename: str = "partial_pcd.ply",
    full_pc_relpath: str = "ptcloud/ptcloud.ply",
    launch_dir: Path | None = None,
) -> None:
    """Run launch/generate_vertices.py to create observation_dict.json and partial point cloud."""
    zed_root = Path(_require_env("ZED_ROOT"))
    launch_dir = launch_dir or Path("conceptgraph/launch")

    result_path = zed_root / scene_name / "pcd_saves" / pkl_filename
    save_path = zed_root / scene_name / "observation_dict.json"
    save_pcd_result_path = zed_root / scene_name / save_partial_pcd_filename
    full_pcd_path = zed_root / scene_name / full_pc_relpath

    cmd: list[str] = [
        "python", str(launch_dir / "generate_vertices.py"),
        "--result_path", str(result_path),
        "--json_file", str(zed_root / scene_name / "sg_cache" / "captions.json"),
        "--save_file", str(save_path),
        "--downsample_ratio", str(downsample_ratio),
        "--no_clip",
        "--save_partial_pcd_path", str(save_pcd_result_path),
        "--full_pc_path", str(full_pcd_path),
    ]
    _exec(cmd)


# --- Orchestration ---

def test_run_all() -> None:
    """Run the default end-to-end sequence analogous to run_all.sh."""
    scene_name = "250820_ECL_500_2_0.01_20_1_test"
    print(f"=== Starting pipeline for scene: {scene_name} ===")
    print("Step 1: extract_2d_classes")
    extract_2d_classes(scene_name)

    print("Step 2: run_map_classes")
    run_map_classes(scene_name)

    print("Step 3: visualize_classes")
    visualize_classes(scene_name)

    print("Step 4: run_scene_graph_new")
    run_scene_graph_new(scene_name)

    print("=== Pipeline complete ===")

