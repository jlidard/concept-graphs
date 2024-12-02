SCENE_NAME=241202_ECL_500_3_0.05_20_1
RESULT_PATH=/home/pbrick/zed2i/$SCENE_NAME/pcd_saves/full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz

python scripts/visualize_cfslam_results.py \
--result_path $RESULT_PATH \
--downsample_ratio 0.01

