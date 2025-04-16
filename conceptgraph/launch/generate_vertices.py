import copy
import json
import os
import pickle
import gzip
import argparse

import matplotlib
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn.functional as F
import open_clip

import distinctipy

from gradslam.structures.pointclouds import Pointclouds

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh
from conceptgraph.slam.utils import filter_objects, merge_objects
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint
import open3d as o3d
from conceptgraph.scripts.visualize_cfslam_results import compute_yaw_aligned_open3d_bbox, create_ball_mesh
from functools import reduce

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--rgb_pcd_path", type=str, default=None)
    parser.add_argument("--bbox_volume_thresh", type=float, default=5e5)

    parser.add_argument("--no_clip", action="store_true",
                        help="If set, the CLIP model will not init for fast debugging.")

    # To inspect the results of merge_overlap_objects
    # This is mainly to quickly try out different thresholds
    parser.add_argument("--merge_overlap_thresh", type=float, default=-1)
    parser.add_argument("--merge_visual_sim_thresh", type=float, default=-1)
    parser.add_argument("--merge_text_sim_thresh", type=float, default=-1)
    parser.add_argument("--obj_min_points", type=int, default=0)
    parser.add_argument("--obj_min_detections", type=int, default=0)
    parser.add_argument("--downsample_ratio", type=float, default=0.05,
                        help="Downsampling ratio for object detections. Lower = more points")
    parser.add_argument("--save_pynt", type=int, default=0, help="If set to 1, the Pynt model will be saved.")
    parser.add_argument("--json_file", type=str, default=None)
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--save_partial_pcd_path", type=str, default=None)
    parser.add_argument("--full_pc_path", type=str, default=None)
    parser.add_argument("--skip_rendering", action="store_true")
    parser.add_argument("--bbox_type", type=str, default="axis_aligned",
                        choices=["axis_aligned", "yaw_aligned"],
                        help="Type of bounding box to compute: 'axis_aligned' or 'yaw_aligned'.")
    return parser


def load_result(result_path):
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if isinstance(results, dict):
        objects = MapObjectList()
        objects.load_serializable(results["objects"])

        if results['bg_objects'] is None:
            bg_objects = None
        else:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results["bg_objects"])

        class_colors = results['class_colors']
    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)

        bg_objects = None
        class_colors = distinctipy.get_colors(len(objects), pastel_factor=0.5)
        class_colors = {str(i): c for i, c in enumerate(class_colors)}
    else:
        raise ValueError("Unknown results type: ", type(results))

    return objects, bg_objects, class_colors


def main(args):
    result_path = args.result_path
    rgb_pcd_path = args.rgb_pcd_path

    assert not (result_path is None and rgb_pcd_path is None), \
        "Either result_path or rgb_pcd_path must be provided."

    if rgb_pcd_path is not None:
        pointclouds = Pointclouds.load_pointcloud_from_h5(rgb_pcd_path)
        global_pcd = pointclouds.open3d(0, include_colors=True)

        if result_path is None:
            print("Only visualizing the pointcloud...")
            o3d.visualization.draw_geometries([global_pcd])
            exit()

    assert not (args.json_file is None), \
    "json_file must be provided."

    object_dict = {}

    objects, bg_objects, class_colors = load_result(result_path)
    pcds = copy.deepcopy(objects.get_values("pcd"))

    # Load edge files and create meshes for the scene graph
    scene_graph_geometries = []
    with open(args.json_file, "r") as f:
        vertices_labels = json.load(f)

    bboxes = []
    
    # Transform point cloud from image frame to x-forward, y-left, z-up frame
    transformation_matrix = np.array([[0, 0, 1, 0],
                                       [-1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 0, 1]])

    # Additional transformation to flip y and z
    flip_yz_matrix = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])

    # Combine the transformations
    transformation_matrix = np.dot(flip_yz_matrix, transformation_matrix)
    for i in range(len(objects)):
        pcd = objects[i]['pcd']

        pcd.transform(transformation_matrix)
        # Remove statistical outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)

        # Keep only inliers
        pcd_inliers = pcd.select_by_index(ind)

        # Visualize the cleaned point cloud
        pcd = pcd_inliers
        if args.bbox_type == "yaw_aligned":
            bbox = compute_yaw_aligned_open3d_bbox(pcd)
            bbox_extent = bbox.extent
        elif args.bbox_type == "axis_aligned":
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox_extent = bbox.get_extent()
        else:
            raise ValueError("Unknown bbox type: ", args.bbox_type)
        width, height, depth = bbox_extent
        volume = width * height * depth  # Volume of a box
        if volume < args.bbox_volume_thresh:
            bboxes.append(bbox)
    object_dict = {}
    # store object id together with bbox extent and quaternion pose of bbox
    for i in range(len(bboxes)):
        # get extent
        bbox = bboxes[i]
        bbox_properties = {}
        if args.bbox_type == "yaw_aligned":
            rot_mat = bbox.R
            bbox_extent = bbox.extent
            bbox_center = bbox.center
        else:
            rot_mat = np.eye(3)
            bbox_extent = bbox.get_extent()
            bbox_center = bbox.get_center()
        # theta = np.arctan2(-rot_mat[2, 0], rot_mat[0, 0])
        theta = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        theta_deg = theta * 180 / np.pi
        bbox_properties["extent"] = bbox_extent
        bbox_properties["center"] = bbox_center
        bbox_properties["theta"] = theta_deg
        if i < len(vertices_labels):
            if "ignore" in vertices_labels[i].lower():
                bbox_properties["semantic_label"] = None
            else:
                bbox_properties["semantic_label"] = vertices_labels[i]
        else:
            bbox_properties["semantic_label"] = None
        object_dict[i] = bbox_properties

    # process all objects and remove those with no semantic label
    keys_to_delete = []
    for k in object_dict.keys():
        semantic_label = object_dict[k]["semantic_label"]
        if semantic_label is None:
            keys_to_delete.append(k)
    for k in keys_to_delete:
        del object_dict[k]
    print(object_dict)

    # save pcd
    pcds_to_merge = []
    for i, pcd in enumerate(pcds):
        pcd.transform(transformation_matrix)
        if i not in keys_to_delete:
            pcds_to_merge.append(pcd)

    if args.save_partial_pcd_path is not None:
        total_pcd = reduce(lambda x, y: x + y, pcds_to_merge)
        o3d.io.write_point_cloud(args.save_partial_pcd_path, total_pcd)

    # save observation
    import pprint
    pprint.pprint(object_dict)

    with open(args.save_file, "wb") as f:
        pickle.dump(object_dict, f)

    # human-readbale
    with open(args.save_file, "w") as f:
        object_dict = {k: {key: (value.tolist() if isinstance(value, np.ndarray) else value) 
                   for key, value in v.items()} 
                   for k, v in object_dict.items()}
        json.dump(object_dict, f, indent=4)

    # rendering
    if args.skip_rendering:
        return

    # Set the title of the window
    vis = o3d.visualization.VisualizerWithKeyCallback()

    if result_path is not None:
        vis.create_window(window_name=f'Open3D - {os.path.basename(result_path)}', width=1280, height=720)
    else:
        vis.create_window(window_name=f'Open3D', width=1280, height=720)

    # Add geometry to the scene
    if args.full_pc_path is not None:
        full_pcd = o3d.io.read_point_cloud(str(args.full_pc_path))
        full_pcd.transform(transformation_matrix)
        vis.add_geometry(full_pcd)
    else:
        # use the segmented pcds
        for geometry in pcds:
            vis.add_geometry(geometry)
    for i, geometry in enumerate(bboxes):
        if i not in keys_to_delete:
            vis.add_geometry(geometry)

    # add origin to the scene
    size = 1
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    vis.add_geometry(frame)

    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background

    main.show_bg_pcd = True

    main.show_global_pcd = False
    main.show_scene_graph = False

    def toggle_scene_graph(vis):
        if args.json_file is None:
            print("No edge file provided.")
            return

        if main.show_scene_graph:
            for geometry in scene_graph_geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in scene_graph_geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)

        main.show_scene_graph = not main.show_scene_graph

    vis.register_key_callback(ord("G"), toggle_scene_graph)

    # Render the scene
    if not args.skip_rendering:
        vis.run()





if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)