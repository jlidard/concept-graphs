#!/bin/bash

ROOT_DIR=/home/jlidard/zed2i
SCENE_NAME=$1
CONFIG_PATH=/home/jlidard/concept-graphs/conceptgraph/dataset/dataconfigs/zed2i/zed2i.yaml
CLASS_SET=ram
PKL_FILENAME=full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz # Change this to the actual output file name of the pkl.gz file

# python scenegraph/build_scenegraph_cfslam_gpt.py \
#     --mode preprocess_images \
#     --masking_option red_outline \
#     --cachedir ${ROOT_DIR}/${SCENE_NAME}/sg_cache \
#     --mapfile ${ROOT_DIR}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pbrick
python ~/predictive_brickwork/src/vlm/extract_node_captions.py \
    --root_dir ${ROOT_DIR}/${SCENE_NAME}/sg_cache \
    --image_dir cfslam_captions_gpt_debug
conda activate cg
