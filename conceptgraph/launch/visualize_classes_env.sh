SCENE_ID=$1
ROOT_DIR=/home/jlidard/pbrick_drive/ECL_buildable/$SCENE_ID
SCENE_NAME=conceptgraphs
RESULT_PATH=$ROOT_DIR/$SCENE_NAME/pcd_saves/full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz

python scripts/visualize_cfslam_results.py \
--result_path $RESULT_PATH \
--downsample_ratio 0.01 \
--no_clip

