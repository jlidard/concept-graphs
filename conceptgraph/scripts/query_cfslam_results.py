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
    topk_similarities, topk_indices = similarities.topk(5)
    query_to_obj_bbox = {text_query: [bboxes[i] for i in topk_indices]}
    query_to_bbox_dict.update(query_to_obj_bbox)

    save_path = result_path.split('/')[:-2]
    save_path = os.path.join('/', *save_path[1:], 'queries')
    os.makedirs(save_path, exist_ok=True)
    for k, v in query_to_bbox_dict.items():
        # dump to .pkl file
        content = [{"rotation": match.R, "center": match.center, "extent": match.extent} for match in v]
        with open(os.path.join(save_path, k + ".pkl"), "wb") as f: pickle.dump(content, f)

    save_to_pyntcloud()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)