SCENE_NAME=250814_ECL_200_2_0.01_20_1
PKL_FILENAME=full_pcd_ram_withbg_allclasses_SINGLETON_post.pkl.gz  # Change this to the actual output file name of the pkl.gz file

python scenegraph/build_scenegraph_cfslam.py \
    --mode extract-node-captions \
    --cachedir ${ZED_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${ZED_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}

python scenegraph/build_scenegraph_cfslam.py \
    --mode refine-node-captions \
    --cachedir ${ZED_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${ZED_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}

python scenegraph/build_scenegraph_cfslam.py \
    --mode build-scenegraph \
    --cachedir ${ZED_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${ZED_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}