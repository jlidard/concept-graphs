SCENE_NAME=241028_ECL_50_5_0.25_20_1

python scripts/generate_vertices.py \
    --edge_file ${ZED_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json \
    --scene_name ${SCENE_NAME} \
    #--save_pynt 0 \
    #--downsample_ratio 0.01