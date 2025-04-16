SCENE_NAME=$1
RESULT_PATH=/home/jlidard/zed2i/$SCENE_NAME/pcd_saves/full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz


python scripts/visualize_cfslam_results.py \
--result_path $RESULT_PATH \
--downsample_ratio 0.01

