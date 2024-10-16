SCENE_NAME=241014_ECL_500_5_0.05_20_1

python scripts/visualize_cfslam_results.py \
    --result_path ${ZED_ROOT}/${SCENE_NAME}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1.pkl.gz \
    --edge_file ${ZED_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json \
    --save_pynt 0 \
    --downsample_ratio 0.01