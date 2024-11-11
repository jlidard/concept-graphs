SCENE_ID=$1
ROOT_DIR=/home/jlidard/pbrick_drive/planters/$SCENE_ID
SCENE_NAME=conceptgraphs
RESULT_PATH=$ROOT_DIR/$SCENE_NAME/pcd_saves/full_pcd_ram_withbg_allclasses_SINGLETON_post.pkl.gz

python scripts/visualize_cfslam_results.py \
--result_path $RESULT_PATH \
--downsample_ratio 0.01

