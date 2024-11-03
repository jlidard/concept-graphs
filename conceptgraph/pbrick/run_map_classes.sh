# On the ConceptGraphs-Detect
SCENE_NAME=lab7
THRESHOLD=1.2

python slam/cfslam_pipeline_batch.py \
    dataset_root=$ZED_ROOT \
    dataset_config=$ZED_CONFIG_PATH \
    stride=5 \
    scene_id=$SCENE_NAME \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.10 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
    dbscan_eps=0.01 \
    dbscan_min_points=1 \
    gsa_variant=ram_withbg_allclasses \
    skip_bg=False \
    max_bbox_area_ratio=5.0 \
    save_suffix=overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1 \
    obj_min_detections=0 \
    dbscan_remove_noise=False \
    downsample_voxel_size=0.01 \
    mask_area_threshold=10 \
    min_points_threshold=8 \
    semantic_threshold=0.1 \
    physical_threshold=0.1 \
    vis_render=False\
    debug_render=False \
    save_objects_all_frames=True
#    filter_interval=20