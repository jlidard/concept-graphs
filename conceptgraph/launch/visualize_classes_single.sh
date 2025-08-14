SCENE_ID=$1
ROOT_DIR=/home/pbrick/dev/data # /$SCENE_ID
SCENE_NAME=250814_ECL_200_2_0.01_20_1
RESULT_PATH=$ROOT_DIR/$SCENE_NAME/pcd_saves/full_pcd_ram_withbg_allclasses_SINGLETON_post.pkl.gz

python scripts/visualize_cfslam_results.py \
--result_path $RESULT_PATH \
--downsample_ratio 0.01 \
--no_clip

