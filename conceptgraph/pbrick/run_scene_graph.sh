SCENE_NAME=241014_ECL_500_5_0.05_20_1
RESULT_FILENAME=full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz

# python scenegraph/build_scenegraph_cfslam.py \
#     --mode extract-node-captions \
#     --cachedir ${ZED_ROOT}/${SCENE_NAME}/sg_cache \
#     --mapfile ${ZED_ROOT}/${SCENE_NAME}/pcd_saves/${RESULT_FILENAME}

python scenegraph/build_scenegraph_cfslam.py \
    --mode refine-node-captions \
    --cachedir ${ZED_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${ZED_ROOT}/${SCENE_NAME}/pcd_saves/${RESULT_FILENAME}

python scenegraph/build_scenegraph_cfslam.py \
    --mode build-scenegraph \
    --cachedir ${ZED_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${ZED_ROOT}/${SCENE_NAME}/pcd_saves/${RESULT_FILENAME}