SCENE_NAME=250814_ECL_200_2_0.01_20_1

python scripts/visualize_cfslam_results.py \
   --result_path ${ZED_ROOT}/${SCENE_NAME}/sg_cache/map/scene_map_cfslam_pruned.pkl.gz \
   --edge_file ${ZED_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json \
   --save_pynt 0

# python scripts/visualize_cfslam_results.py \
#     --result_path ${ZED_ROOT}/${SCENE_NAME}/pcd_saves/full_pcd_ram_withbg_allclasses_SINGLETON_post.pkl.gz \
#     --edge_file ${ZED_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json \
#     --save_pynt 0\
#     --downsample_ratio 0.001