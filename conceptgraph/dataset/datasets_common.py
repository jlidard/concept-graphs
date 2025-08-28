"""
PyTorch dataset classes for datasets in the NICE-SLAM format.
Large chunks of code stolen and adapted from:
https://github.com/cvg/nice-slam/blob/645b53af3dc95b4b348de70e759943f7228a61ca/src/utils/datasets.py

"""

import glob
import os
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import yaml
from natsort import natsorted

from gradslam.datasets import datautils
from gradslam.geometry.geometryutils import relative_transformation

from conceptgraph.utils.general_utils import to_scalar, measure_time


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

def from_intrinsics_matrix(K: torch.Tensor) -> tuple[float, float, float, float]:
    '''
    Get fx, fy, cx, cy from the intrinsics matrix
    
    return 4 scalars
    '''
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy


class GradSLAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True, # If True, the pose is relative to the first frame
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]

        self.dtype = dtype

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError(
                "end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start)
            )

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )
        self.crop_size = (
            config_dict["camera_params"]["crop_size"]
            if "crop_size" in config_dict["camera_params"]
            else None
        )

        self.crop_edge = None
        if "crop_edge" in config_dict["camera_params"].keys():
            self.crop_edge = config_dict["camera_params"]["crop_edge"]

        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")
        if self.load_embeddings:
            if len(self.color_paths) != len(self.embedding_paths):
                raise ValueError(
                    "Mismatch between number of color images and number of embedding files."
                )
        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()
        
        if self.end == -1:
            self.end = self.num_imgs

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]
        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]
        # Update self.num_images after subsampling the dataset
        self.num_imgs = len(self.color_paths)

        # self.transformed_poses = datautils.poses_to_transforms(self.poses)
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self):
        return self.num_imgs

    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError

    def load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.png_depth_scale
    
    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )
        
    def get_cam_K(self):
        '''
        Return camera intrinsics matrix K
        
        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        '''
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K
    
    def read_embedding_from_file(self, embedding_path: str):
        '''
        Read embedding from file and process it. To be implemented in subclass for each dataset separately.
        '''
        raise NotImplementedError

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        if ".png" in depth_path:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".npy" in depth_path:
            depth = np.load(depth_path)
        else:
            raise NotImplementedError

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = datautils.scale_intrinsics(
            K, self.height_downsample_ratio, self.width_downsample_ratio
        )
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )


class ICLDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: Dict,
        basedir: Union[Path, str],
        sequence: Union[Path, str],
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[Union[Path, str]] = "embeddings",
        embedding_dim: Optional[int] = 512,
        embedding_file_extension: Optional[str] = "pt",
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # Attempt to find pose file (*.gt.sim)
        self.pose_path = glob.glob(os.path.join(self.input_folder, "*.gt.sim"))
        if self.pose_path == 0:
            raise ValueError("Need pose file ending in extension `*.gt.sim`")
        self.pose_path = self.pose_path[0]
        self.embedding_file_extension = embedding_file_extension
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(
                    f"{self.input_folder}/{self.embedding_dir}/*.{self.embedding_file_extension}"
                )
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []

        lines = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()

        _posearr = []
        for line in lines:
            line = line.strip().split()
            if len(line) == 0:
                continue
            _npvec = np.asarray(
                [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
            )
            _posearr.append(_npvec)
        _posearr = np.stack(_posearr)

        for pose_line_idx in range(0, _posearr.shape[0], 3):
            _curpose = np.zeros((4, 4))
            _curpose[3, 3] = 3
            _curpose[0] = _posearr[pose_line_idx]
            _curpose[1] = _posearr[pose_line_idx + 1]
            _curpose[2] = _posearr[pose_line_idx + 2]
            poses.append(torch.from_numpy(_curpose).float())

        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)



class ZedDataset(GradSLAMDataset):

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # only poses/images/depth corresponding to the realsense_camera_order are read/used
        self.pose_path = os.path.join(self.input_folder, 'zed_poses')
        print(self.pose_path)
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/filt_depth*.npy"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(os.path.join(self.pose_path, "*.npy")))
        poses = []
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
        for posefile in posefiles:
            c2w = torch.from_numpy(np.load(posefile)).float()
            _R = c2w[:3, :3]
            _t = c2w[:3, 3]
            _pose = c2w @ P
            poses.append(_pose)
        return poses


class ZedDatasetDynamicIntrinsics(ZedDataset):


    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)

        # scene_intrinsic_path = os.path.join(self.input_folder, "intrinsics", "intrinsics.npy")
        # scene_intrinsic = np.load(scene_intrinsic_path)
        # config_dict['camera_params']['fx'] = scene_intrinsic[0, 0]
        # config_dict['camera_params']['fy'] = scene_intrinsic[1, 1]
        # config_dict['camera_params']['cx'] = scene_intrinsic[0, 2]
        # config_dict['camera_params']['cy'] = scene_intrinsic[1, 2]
        #
        # # set image width and height
        # config_dict['camera_params']['image_width'] = scene_intrinsic[0, 2] * 2
        # config_dict['camera_params']['image_height'] = scene_intrinsic[1, 2] * 2

        # only poses/images/depth corresponding to the realsense_camera_order are read/used
        self.pose_path = os.path.join(self.input_folder, 'poses')
        super().__init__(
            config_dict,
            basedir,
            sequence,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/*.npy"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

class ZedDatasetPreprocess(GradSLAMDataset):

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # only poses/images/depth corresponding to the realsense_camera_order are read/used
        self.pose_path = os.path.join(self.input_folder, 'poses')
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.npy"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        # load pose files from json
        import json
        import re


        # Function to extract the frame number from the file path (e.g., frame_00001.jpg -> 1)
        def get_frame_number(file_path):
            match = re.search(r'frame_(\d+)', file_path)
            return int(match.group(1)) if match else 0

        # Load the JSON file
        json_file_path = os.path.join(self.input_folder, 'transforms.json')
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Sort frames by frame number
        sorted_frames = sorted(data['frames'], key=lambda frame: get_frame_number(frame['file_path']))

        # Extract the list of poses (transform_matrix)
        poses = [torch.Tensor(frame['transform_matrix']) for frame in sorted_frames]
        # P = torch.tensor(
        #     [
        #         [1, 0, 0, 0],
        #         [0, 0, 1, 0],
        #         [0, -1, 0, 0],
        #         [0, 0, 0, 1]
        #     ]
        # ).float()
        for i in range(len(poses)):
            # poses[i] = torch.linalg.inv(P) @ poses[i]
            poses[i][:3, :3] = torch.eye(3)

        # Display the first pose as an example
        print('Loaded poses like example:')
        print(poses[0])

        return poses

        
def load_dataset_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_dataset_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def common_dataset_to_batch(dataset):
    colors, depths, poses = [], [], []
    intrinsics, embeddings = None, None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose, _embedding = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
        if _embedding is not None:
            if embeddings is None:
                embeddings = [_embedding]
            else:
                embeddings.append(_embedding)
    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    if embeddings is not None:
        embeddings = torch.stack(embeddings, dim=1)
        # # (1, NUM_IMG, DIM_EMBED, H, W) -> (1, NUM_IMG, H, W, DIM_EMBED)
        # embeddings = embeddings.permute(0, 1, 3, 4, 2)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()
    if embeddings is not None:
        embeddings = embeddings.float()
    return colors, depths, intrinsics, poses, embeddings

@measure_time
def get_dataset(dataconfig, basedir, sequence, **kwargs):
    config_dict = load_dataset_config(dataconfig)
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["zed2i"]:
        return ZedDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["zed2ipre"]:
        return ZedDatasetPreprocess(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["zed2i-dynamic-intrinsics"]:
        return ZedDatasetDynamicIntrinsics(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")
