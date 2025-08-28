#!/bin/bash

SCENE_NAME=$1
CLASS_SET=ram
PKL_FILENAME=full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz # Change this to the actual output file name of the pkl.gz file

python scenegraph/build_scenegraph_cfslam_gpt.py \
    --mode preprocess_images \
    --masking_option red_outline \
    --cachedir $ZED_ROOT/${SCENE_NAME}/sg_cache \
    --mapfile $ZED_ROOT/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}

python scripts/extract_node_captions.py \
    --root_dir $ZED_ROOT/${SCENE_NAME}/sg_cache \
    --image_dir cfslam_captions_gpt_debug
