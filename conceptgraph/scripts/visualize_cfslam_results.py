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


def compute_yaw_aligned_open3d_bbox(point_cloud):
    """
    Computes a yaw-aligned Open3D bounding box for a 3D point cloud with [x, z, y] ordering.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): Open3D PointCloud object with points in [x, z, y] order.

    Returns:
        o3d.geometry.OrientedBoundingBox: An Open3D OrientedBoundingBox object for the 3D bounding box.
    """
    # Convert Open3D point cloud to NumPy array
    points = np.asarray(point_cloud.points)

    # Project points to the XY plane (which is [x, y] in the [x, z, y] format)
    points_xy = points[:, [0, 2]]

    # Compute 2D convex hull
    hull = ConvexHull(points_xy)
    hull_points = points_xy[hull.vertices]

    # Find minimum area bounding rectangle
    hull_polygon = MultiPoint(hull_points).convex_hull
    min_area_rect = hull_polygon.minimum_rotated_rectangle

    # Extract rectangle coordinates from the 2D bounding rectangle
    bbox_2d_coords = np.array(min_area_rect.exterior.coords)[:-1]  # Skip the repeated last point

    # Get min and max Z (vertical axis)
    z_min = np.min(points[:, 1])
    z_max = np.max(points[:, 1])

    # Define the 8 corners of the 3D bounding box based on the 2D rectangle and z bounds
    bbox_3d_corners = np.array([
        [bbox_2d_coords[0][0], z_min, bbox_2d_coords[0][1]],
        [bbox_2d_coords[1][0], z_min, bbox_2d_coords[1][1]],
        [bbox_2d_coords[2][0], z_min, bbox_2d_coords[2][1]],
        [bbox_2d_coords[3][0], z_min, bbox_2d_coords[3][1]],
        [bbox_2d_coords[0][0], z_max, bbox_2d_coords[0][1]],
        [bbox_2d_coords[1][0], z_max, bbox_2d_coords[1][1]],
        [bbox_2d_coords[2][0], z_max, bbox_2d_coords[2][1]],
        [bbox_2d_coords[3][0], z_max, bbox_2d_coords[3][1]]
    ])

    # Create an Open3D OrientedBoundingBox from the corner points
    bbox_open3d = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_3d_corners))

    return bbox_open3d


# Example usage
# pcd = o3d.io.read_point_cloud("path_to_your_point_cloud.ply")  # Replace with actual point cloud data
# bbox_open3d = compute_yaw_aligned_open3d_bbox(pcd)
# o3d.visualization.draw_geometries([pcd, bbox_open3d])


def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """
    Create a colored mesh sphere.
    
    Args:
    - center (tuple): (x, y, z) coordinates for the center of the sphere.
    - radius (float): Radius of the sphere.
    - color (tuple): RGB values in the range [0, 1] for the color of the sphere.
    
    Returns:
    - o3d.geometry.TriangleMesh: Colored mesh sphere.
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--rgb_pcd_path", type=str, default=None)
    parser.add_argument("--edge_file", type=str, default=None)
    parser.add_argument("--bbox_volume_thresh", type=float, default=5e3)
    
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

from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm

def compute_iou_3d(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) for two 3D bounding boxes using ConvexHull.
    Args:
        bbox1 (o3d.geometry.OrientedBoundingBox): First bounding box.
        bbox2 (o3d.geometry.OrientedBoundingBox): Second bounding box.
    Returns:
        float: IoU value.
    """
    # Get corner points for both bounding boxes
    corners1 = np.asarray(bbox1.get_box_points())
    corners2 = np.asarray(bbox2.get_box_points())

    try:
        # Combine the points for the intersection volume calculation
        combined_points = np.vstack((corners1, corners2))
        intersection_hull = ConvexHull(combined_points)
        
        # Check if the combined hull volume equals the sum of individual volumes
        intersection_volume = intersection_hull.volume
        if np.isclose(intersection_volume, bbox1.volume() + bbox2.volume()):
            # If no intersection, the combined hull volume is just the sum of individual volumes
            intersection_volume = 0.0

    except Exception as e:
        # Handle cases where no intersection is computable
        print(f"Warning: Could not compute intersection: {e}")
        intersection_volume = 0.0

    # Calculate union volume
    union_volume = bbox1.volume() + bbox2.volume() - intersection_volume

    return intersection_volume / union_volume if union_volume > 0 else 0


def merge_objects_based_on_iou(objects, iou_threshold):
    """
    Merge objects based on IoU of their yaw-aligned oriented bounding boxes.
    Args:
        objects (MapObjectList): List of objects with 'pcd' and 'bbox' attributes.
        iou_threshold (float): IoU threshold for merging.
    Returns:
        MapObjectList: Updated object list with merged objects.
    """
    merged_objects = []
    used_indices = set()

    for i in tqdm(range(len(objects))):
        if i in used_indices:
            continue
        obj_i = objects[i]
        bbox_i = compute_yaw_aligned_open3d_bbox(obj_i['pcd'])

        merged_pcd_points = np.asarray(obj_i['pcd'].points)
        merged_indices = {i}

        for j in range(i + 1, len(objects)):
            if j in used_indices:
                continue
            obj_j = objects[j]
            bbox_j = compute_yaw_aligned_open3d_bbox(obj_j['pcd'])
            iou = compute_iou_3d(bbox_i, bbox_j)
            print(iou)

            if iou >= iou_threshold:
                used_indices.add(j)
                merged_indices.add(j)
                merged_pcd_points = np.vstack((merged_pcd_points, np.asarray(obj_j['pcd'].points)))

        # Create a new object from merged points
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_pcd_points)

        # Handle missing keys with default values
        merged_obj = {
            'pcd': merged_pcd,
            'class_id': np.concatenate([objects[idx].get('class_id', []) for idx in merged_indices]),
            'detections': sum(objects[idx].get('detections', 0) for idx in merged_indices)  # Default to 0 if missing
        }
        merged_objects.append(merged_obj)
        used_indices.update(merged_indices)

    return MapObjectList(merged_objects)


# Integrate the merge function into your pipeline
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
        
    objects, bg_objects, class_colors = load_result(result_path)
    
    # Perform object merging based on IoU
    iou_merge_thresh = 0
    if iou_merge_thresh > 0:
        print(f"Merging objects with IoU threshold {iou_merge_thresh}...")
        objects = merge_objects_based_on_iou(objects, iou_merge_thresh)
        print("Merging completed.")
    

    assert not (result_path is None and rgb_pcd_path is None), \
        "Either result_path or rgb_pcd_path must be provided."

    if rgb_pcd_path is not None:        
        pointclouds = Pointclouds.load_pointcloud_from_h5(rgb_pcd_path)
        global_pcd = pointclouds.open3d(0, include_colors=True)
        
        if result_path is None:
            print("Only visualizing the pointcloud...")
            o3d.visualization.draw_geometries([global_pcd])
            exit()
        
    objects, bg_objects, class_colors = load_result(result_path)
    
    if args.edge_file is not None:
        # Load edge files and create meshes for the scene graph
        scene_graph_geometries = []
        with open(args.edge_file, "r") as f:
            edges = json.load(f)

        classes = objects.get_most_common_class()
        colors = [class_colors[str(c)] for c in classes]
        obj_centers = []
        for obj, c in zip(objects, colors):
            pcd = obj['pcd']
            bbox = obj['bbox']
            points = np.asarray(pcd.points)
            center = np.mean(points, axis=0)
            extent = bbox.get_max_bound()
            extent = np.linalg.norm(extent)
            # radius = extent ** 0.5 / 25
            radius = 0.10
            obj_centers.append(center)

            # remove the nodes on the ceiling, for better visualization
            ball = create_ball_mesh(center, radius, c)
            scene_graph_geometries.append(ball)
            
        for edge in edges:
            if edge['object_relation'] == "none of these":
                continue
            id1 = edge["object1"]['id']
            id2 = edge["object2"]['id']

            line_mesh = LineMesh(
                points = np.array([obj_centers[id1], obj_centers[id2]]),
                lines = np.array([[0, 1]]),
                colors = [1, 0, 0],
                radius=0.02
            )

            scene_graph_geometries.extend(line_mesh.cylinder_segments)
    
    if not args.no_clip:
        print("Initializing CLIP model...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to("cuda")
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        print("Done initializing CLIP model.")

    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    if bg_objects is not None:
        indices_bg = np.arange(len(objects), len(objects) + len(bg_objects))
        objects.extend(bg_objects)
        
    # Sub-sample the point cloud for better interactive experience
    for i in range(len(objects)):
        pcd = objects[i]['pcd']
        pcd = pcd.voxel_down_sample(args.downsample_ratio)
        objects[i]['pcd'] = pcd
    
    pcds = copy.deepcopy(objects.get_values("pcd"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))

    bboxes = []

    for i in range(len(objects)):
        pcd = objects[i]['pcd']
        bbox = compute_yaw_aligned_open3d_bbox(pcd)
        bbox_extent = bbox.extent
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
        rot_mat = bbox.R
        theta = np.arctan2(-rot_mat[2, 0], rot_mat[0, 0])
        theta_deg = theta * 180 / np.pi
        bbox_properties["extent"] = bbox.extent
        bbox_properties["center"] = bbox.center
        bbox_properties["theta"] = theta_deg
        bbox_properties["semantic_label"] = None
        object_dict[i] = bbox_properties

        # # get axis aligned bounding box
        # aabb = pcd.get_axis_aligned_bounding_box()
        # aabb.color = (1, 0, 0)  # Set the bounding box color to red
        # bboxes.append(aabb)

    # Get the color for each object when colored by their class
    object_classes = []
    for i in range(len(objects)):
        obj = objects[i]
        pcd = pcds[i]
        obj_classes = np.asarray(obj['class_id'])
        # Get the most common class for this object as the class
        values, counts = np.unique(obj_classes, return_counts=True)
        obj_class = values[np.argmax(counts)]
        object_classes.append(obj_class)
    
    # Set the title of the window
    vis = o3d.visualization.VisualizerWithKeyCallback()

    if result_path is not None:
        vis.create_window(window_name=f'Open3D - {os.path.basename(result_path)}', width=1280, height=720)
    else:
        vis.create_window(window_name=f'Open3D', width=1280, height=720)

    # Add geometry to the scene
    for geometry in pcds + bboxes:
        vis.add_geometry(geometry)

    # add origin to the scene
    size = 1
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    vis.add_geometry(frame)

    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background

    def save_to_pyntcloud():
        from pyntcloud import PyntCloud

        scene_id = result_path.split('/')[-3]

        # Assuming 'point_cloud' is your PointCloud object
        # Extract XYZ points
        xyz = np.concatenate([pcds[i].points for i in range(len(pcds)) if i not in indices_bg])

        # Extract RGB colors (if available)
        rgb = np.concatenate([pcds[i].colors for i in range(len(pcds)) if i not in indices_bg])

        # Concatenate XYZ and RGB into a single array
        # RGB values should be scaled to 0-255 if needed
        features = np.hstack((xyz, rgb))

        # Create a DataFrame for PyntCloud
        df = pd.DataFrame(features, columns=['x', 'y', 'z', 'r', 'g', 'b'])

        # Convert to PyntCloud object
        pyntcloud = PyntCloud(df)

        # Save the point cloud as a .ply file
        pyntcloud.to_file(os.path.join('/home/jlidard/predictive_brickwork/tests/perception', f"{scene_id}.ply"))

    if args.save_pynt == 1:
        save_to_pyntcloud()

    query_to_bbox_dict = {}

    main.show_bg_pcd = True

    def toggle_bg_pcd(vis):
        if bg_objects is None:
            print("No background objects found.")
            return
        
        for idx in indices_bg:
            if main.show_bg_pcd:
                vis.remove_geometry(pcds[idx], reset_bounding_box=False)
                vis.remove_geometry(bboxes[idx], reset_bounding_box=False)
            else:
                vis.add_geometry(pcds[idx], reset_bounding_box=False)
                vis.add_geometry(bboxes[idx], reset_bounding_box=False)
        
        main.show_bg_pcd = not main.show_bg_pcd
        
    main.show_global_pcd = False
    def toggle_global_pcd(vis):
        if args.rgb_pcd_path is None:
            print("No RGB pcd path provided.")
            return
        
        if main.show_global_pcd:
            vis.remove_geometry(global_pcd, reset_bounding_box=False)
        else:
            vis.add_geometry(global_pcd, reset_bounding_box=False)
        
        main.show_global_pcd = not main.show_global_pcd
        
    main.show_scene_graph = False
    def toggle_scene_graph(vis):
        if args.edge_file is None:
            print("No edge file provided.")
            return
        
        if main.show_scene_graph:
            for geometry in scene_graph_geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in scene_graph_geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)
        
        main.show_scene_graph = not main.show_scene_graph
        
    def color_by_class(vis):
        for i in range(len(objects)):
            pcd = pcds[i]
            obj_class = object_classes[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    class_colors[str(obj_class)],
                    (len(pcd.points), 1)
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_rgb(vis):
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = objects[i]['pcd'].colors
        
        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_instance(vis):
        instance_colors = cmap(np.linspace(0, 1, len(pcds)))
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    instance_colors[i, :3],
                    (len(pcd.points), 1)
                )
            )
            
        for pcd in pcds:
            vis.update_geometry(pcd)
        
    def color_by_clip_sim(vis):
        if args.no_clip:
            print("CLIP model is not initialized.")
            return

        text_query = input("Enter your query: ")
        text_queries = [text_query]
        
        text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
        text_query_ft = clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()
        
        # similarities = objects.compute_similarities(text_query_ft)
        objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        similarities = F.cosine_similarity(
            text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
        )
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]

        # get the top 5 similar objects and store as a dict from query to list of indices
        topk_similarities, topk_indices = similarities.topk(20)

        for i in range(len(objects)):
            pcd = pcds[i]
            map_colors = np.asarray(pcd.colors)

            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    [
                        similarity_colors[i, 0].item(),
                        similarity_colors[i, 1].item(),
                        similarity_colors[i, 2].item()
                    ],
                    (len(pcd.points), 1)
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def save_view_params(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("temp.json", param)
        
    vis.register_key_callback(ord("B"), toggle_bg_pcd)
    vis.register_key_callback(ord("S"), toggle_global_pcd)
    vis.register_key_callback(ord("C"), color_by_class)
    vis.register_key_callback(ord("R"), color_by_rgb)
    vis.register_key_callback(ord("F"), color_by_clip_sim)
    vis.register_key_callback(ord("I"), color_by_instance)
    vis.register_key_callback(ord("V"), save_view_params)
    vis.register_key_callback(ord("G"), toggle_scene_graph)
    
    # Render the scene
    vis.run()
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)