"""
Build a scene graph from the segment-based map and captions from LLaVA.
"""

import gc
import gzip
import json
import os
import pickle as pkl
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, Union
from textwrap import wrap
from conceptgraph.utils.general_utils import prjson

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import rich
import torch
import tyro
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from tqdm import tqdm, trange
from transformers import logging as hf_logging

import sys
from pathlib import Path

# Add the directory containing the file to the Python path
file_path = Path("/home/jlidard/predictive_brickwork/src/vlm/gpt4o_interface.py")
sys.path.append(str(file_path.parent))

# from mappingutils import (
#     MapObjectList,
#     compute_3d_giou_accuracte_batch,
#     compute_3d_iou_accuracte_batch,
#     compute_iou_batch,
#     compute_overlap_matrix_faiss,
#     num_points_closer_than_threshold_batch,
# )

torch.autograd.set_grad_enabled(False)
hf_logging.set_verbosity_error()

# Import OpenAI API
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class ProgramArgs:
    mode: Literal[
        "preprocess_images"
    ]

    # Path to cache directory
    cachedir: str = "saved/room0"
    
    prompts_path: str = "prompts/gpt_prompts.json"

    # Path to map file
    mapfile: str = "saved/room0/map/scene_map_cfslam.pkl.gz"

    # Device to use
    device: str = "cuda:0"

    # Voxel size for downsampling
    downsample_voxel_size: float = 0.025

    # Maximum number of detections to consider, per object
    max_detections_per_object: int = 10

    # Suppress objects with less than this number of observations
    min_views_per_object: int = 2

    # List of objects to annotate (default: all objects)
    annot_inds: Union[List[int], None] = None

    # Masking option
    masking_option: Literal["blackout", "red_outline", "none"] = "none"

def load_scene_map(args, scene_map):
    """
    Loads a scene map from a gzip-compressed pickle file. This is a function because depending whether the mapfile was made using cfslam_pipeline_batch.py or merge_duplicate_objects.py, the file format is different (see below). So this function handles that case.
    
    The function checks the structure of the deserialized object to determine
    the correct way to load it into the `scene_map` object. There are two
    expected formats:
    1. A dictionary containing an "objects" key.
    2. A list or a dictionary (replace with your expected type).
    """
    
    with gzip.open(Path(args.mapfile), "rb") as f:
        loaded_data = pkl.load(f)
        
        # Check the type of the loaded data to decide how to proceed
        if isinstance(loaded_data, dict) and "objects" in loaded_data:
            scene_map.load_serializable(loaded_data["objects"])
        elif isinstance(loaded_data, list) or isinstance(loaded_data, dict):  # Replace with your expected type
            scene_map.load_serializable(loaded_data)
        else:
            raise ValueError("Unexpected data format in map file.")
        print(f"Loaded {len(scene_map)} objects")



def crop_image_pil(image: Image, x1: int, y1: int, x2: int, y2: int, padding: int = 0) -> Image:
    """
    Crop the image with some padding

    Args:
        image: PIL image
        x1, y1, x2, y2: bounding box coordinates
        padding: padding around the bounding box

    Returns:
        image_crop: PIL image

    Implementation from the CFSLAM repo
    """
    image_width, image_height = image.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)

    image_crop = image.crop((x1, y1, x2, y2))
    return image_crop


def draw_red_outline(image, mask):
    """ Draw a red outline around the object i nan image"""
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    red_outline = [255, 0, 0]

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red outlines around the object. The last argument "3" indicates the thickness of the outline.
    cv2.drawContours(image_np, contours, -1, red_outline, 3)

    # # Optionally, add padding around the object by dilating the drawn contours
    # kernel = np.ones((5, 5), np.uint8)
    # image_np = cv2.dilate(image_np, kernel, iterations=1)

    image_pil = Image.fromarray(image_np)

    return image_pil


def crop_image_and_mask(image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """ Crop the image and mask with some padding. I made a single function that crops both the image and the mask at the same time because I was getting shape mismatches when I cropped them separately.This way I can check that they are the same shape."""

    image = np.array(image)
    # Verify initial dimensions
    if image.shape[:2] != mask.shape:
        raise ValueError("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))


    # Define the cropping coordinates
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    # round the coordinates to integers
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    # Crop the image and the mask
    image_crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # Verify cropped dimensions
    if image_crop.shape[:2] != mask_crop.shape:
        print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
        return None, None

    # convert the image back to a pil image
    image_crop = Image.fromarray(image_crop)

    return image_crop, mask_crop

def blackout_nonmasked_area(image_pil, mask):
    """ Blackout the non-masked area of an image"""
    # convert image to numpy array
    image_np = np.array(image_pil)
    # Create an all-black image of the same shape as the input image
    black_image = np.zeros_like(image_np)
    # Wherever the mask is True, replace the black image pixel with the original image pixel
    black_image[mask] = image_np[mask]
    # convert back to pil image
    black_image = Image.fromarray(black_image)
    return black_image

def plot_images_with_captions(images, captions, confidences, low_confidences, masks, savedir, idx_obj):
    """ This is debug helper function that plots the images with the captions and masks overlaid and saves them to a directory. This way you can inspect exactly what the LLaVA model is captioning which image with the mask, and the mask confidence scores overlaid."""

    n = min(9, len(images))  # Only plot up to 9 images
    nrows = int(np.ceil(n / 3))
    ncols = 3 if n > 1 else 1
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)  # Adjusted figsize

    for i in range(n):
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        ax.imshow(images[i])

        # Apply the mask to the image
        img_array = np.array(images[i])
        # if img_array.shape[:2] != masks[i].shape:
        #     ax.text(0.5, 0.5, "Plotting error: Shape mismatch between image and mask", ha='center', va='center')
        # else:
        #     green_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
        #     green_mask[masks[i]] = [0, 255, 0]  # Green color where mask is True
        #     ax.imshow(green_mask, alpha=0.15)  # Overlay with transparency

        title_text = f"Caption: {captions[i]}\nConfidence: {confidences[i]:.2f}"
        if low_confidences[i]:
            title_text += "\nLow Confidence"

        # Wrap the caption text
        wrapped_title = '\n'.join(wrap(title_text, 30))

        # ax.set_title(wrapped_title, fontsize=12)  # Reduced font size for better fitting
        ax.axis('off')

    # Remove any unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis('off')

    plt.tight_layout()
    plt.savefig(savedir / f"{idx_obj}.png")
    plt.close()



def preprocess_images(args):
    from conceptgraph.llava.llava_model import LLaVaChat

    # NOTE: args.mapfile is in cfslam format
    from conceptgraph.slam.slam_classes import MapObjectList

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)

    # Scene map is in CFSLAM format
    # keys: 'image_idx', 'mask_idx', 'color_path', 'class_id', 'num_detections',
    # 'mask', 'xyxy', 'conf', 'n_points', 'pixel_area', 'contain_number', 'clip_ft',
    # 'text_ft', 'pcd_np', 'bbox_np', 'pcd_color_np'

    # Imports to help with feature extraction
    # from extract_mask_level_features import (
    #     crop_bbox_from_img,
    #     get_model_and_preprocessor,
    #     preprocess_and_encode_pil_image,
    # )

    # Creating a namespace object to pass args to the LLaVA chat object
    chat_args = SimpleNamespace()
    chat_args.model_path = os.getenv("LLAVA_CKPT_PATH")
    chat_args.conv_mode = "v0_mmtag" # "multimodal"
    chat_args.num_gpus = 1

    # rich console for pretty printing
    console = rich.console.Console()

    # Initialize LLaVA chat
    # chat = LLaVaChat(chat_args.model_path, chat_args.conv_mode, chat_args.num_gpus)
    # chat = LLaVaChat(chat_args)
    # print("LLaVA chat initialized...")
    query = "Describe the object enclosed by the red outline. F or brick structures and any plants they contain, say brick structure."

    # system_prompt = ("You are a construction robot looking at some images to determine their key subjects. "
    #                  "Answer the prompts to the best of yor ability in 1-4 works max.")
    # model = GPT4OInterface(model=model_str, system_prompt=system_prompt, max_tokens=1, return_logprobs=True)  # mcqa
    #
    # for i, file_name in enumerate(sorted(os.listdir(file_path))):
    #     # Process only image files
    #     if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
    #         image_path = os.path.join(file_path, file_name)
    #         image_paths.append(image_path)
    # result, key_logprobs = model.query(user_prompt, image_path=image_paths)
    # query = "Describe the object in the image that is outlined in red."

    # Directories to save features and captions
    savedir_feat = Path(args.cachedir) / "cfslam_feat_llava"
    savedir_feat.mkdir(exist_ok=True, parents=True)
    savedir_captions = Path(args.cachedir) / "cfslam_captions_gpt"
    savedir_captions.mkdir(exist_ok=True, parents=True)
    savedir_debug = Path(args.cachedir) / "cfslam_captions_gpt_debug"
    savedir_debug.mkdir(exist_ok=True, parents=True)

    caption_dict_list = []

    for idx_obj, obj in tqdm(enumerate(scene_map), total=len(scene_map)):
        conf = obj["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]

        features = []
        captions = []
        low_confidences = []

        image_list = []
        caption_list = []
        confidences_list = []
        low_confidences_list = []
        mask_list = []  # New list for masks
        # if len(idx_most_conf) < 2:
        #     continue
        idx_most_conf = idx_most_conf[:args.max_detections_per_object]

        for idx_det in tqdm(idx_most_conf):
            # image = Image.open(correct_path).convert("RGB")
            image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            xyxy = obj["xyxy"][idx_det]
            class_id = obj["class_id"][idx_det]
            # Retrieve and crop mask
            mask = obj["mask"][idx_det]

            padding = 100
            x1, y1, x2, y2 = xyxy
            # image_crop = crop_image_pil(image, x1, y1, x2, y2, padding=padding)
            image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
            if args.masking_option == "blackout":
                image_crop_modified = blackout_nonmasked_area(image_crop, mask_crop)
            elif args.masking_option == "red_outline":
                image_crop_modified = draw_red_outline(image_crop, mask_crop)
            else:
                image_crop_modified = image_crop  # No modification

            _w, _h = image_crop.size
            if _w * _h < 50 * 50:
                # captions.append("small object")
                print("small object. Skipping LLaVA captioning...")
                low_confidences.append(True)
                continue
            else:
                low_confidences.append(False)

            # image_tensor = chat.image_processor.preprocess(image_crop, return_tensors="pt")["pixel_values"][0]
            # image_tensor = chat.image_processor.preprocess(image_crop_modified, return_tensors="pt")["pixel_values"][0]
            #
            # image_features = chat.encode_image(image_tensor[None, ...].half().cuda())
            # features.append(image_features.detach().cpu())
            #
            # chat.reset()
            # console.print("[bold red]User:[/bold red] " + query)
            # outputs = chat(query=query, image_features=image_features)
            # console.print("[bold green]LLaVA:[/bold green] " + outputs)
            # captions.append(outputs)

            # print(f"Line 274, obj['mask'][idx_det].shape: {obj['mask'][idx_det].shape}")
            # print(f"Line 276, image.size: {image.size}")

            # For the LLava debug folder
            outputs = ""
            conf_value = conf[idx_det]
            image_list.append(image_crop_modified)
            caption_list.append(outputs)
            confidences_list.append(conf_value)
            low_confidences_list.append(low_confidences[-1])
            # mask_list.append(mask_crop)  # Add the cropped mask

        caption_dict_list.append(
            {
                "id": idx_obj,
                # "captions": captions,
                # "low_confidences": low_confidences,
            }
        )

        # Concatenate the features
        if len(features) > 0:
            features = torch.cat(features, dim=0)

        # Save the feature descriptors
        torch.save(features, savedir_feat / f"{idx_obj}.pt")
        
        # Again for the LLava debug folder
        if len(image_list) > 0:
            plot_images_with_captions(image_list, caption_list, confidences_list, low_confidences_list, mask_list, savedir_debug, idx_obj)



    # Save the captions
    # Remove the "The central object in the image is " prefix from 
    # the captions as it doesnt convey and actual info
    # for item in caption_dict_list:
    #     item["captions"] = [caption.replace("The central object in the image is ", "") for caption in item["captions"]]
    # Save the captions to a json file
    with open(Path(args.cachedir) / "cfslam_llava_captions.json", "w", encoding="utf-8") as f:
        json.dump(caption_dict_list, f, indent=4, sort_keys=False)


def save_json_to_file(json_str, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_str, f, indent=4, sort_keys=False)



def extract_object_tag_from_json_str(json_str):
    start_str_found = False
    is_object_tag = False
    object_tag_complete = False
    object_tag = ""
    r = json_str.strip().split()
    for _idx, _r in enumerate(r):
        if not start_str_found:
            # Searching for open parenthesis of JSON
            if _r == "{":
                start_str_found = True
                continue
            else:
                continue
        # Start string found. Now skip everything until the object_tag field
        if not is_object_tag:
            if _r == '"object_tag":':
                is_object_tag = True
                continue
            else:
                continue
        # object_tag field found. Read it
        if is_object_tag and not object_tag_complete:
            if _r == '"':
                continue
            else:
                if _r.strip() in [",", "}"]:
                    break
                object_tag += f" {_r}"
                continue
    return object_tag


def main():
    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)

    # print using masking option
    print(f"args.masking_option: {args.masking_option}")

    if args.mode == "preprocess_images":
        preprocess_images(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
