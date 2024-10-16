RESULT_PATH=/home/pbrick/zed2i/241014_ECL_500_5_0.05_20_1/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz

python scripts/visualize_cfslam_results.py \
--result_path $RESULT_PATH \
--downsample_ratio 0.05

