ROOT_DIR='/home/jlidard/pbrick_drive/planters/brick_001'
SCENE_NAME=conceptgraphs
RESULT_PATH=$ROOT_DIR/$SCENE_NAME/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1.pkl.gz

python scripts/visualize_cfslam_results.py \
--result_path $RESULT_PATH \
--downsample_ratio 0.01

