# On the ConceptGraphs-Detect
ROOT_DIR='/home/jlidard/pbrick_drive/planters/brick_001'
CONFIG_PATH='/home/jlidard/pbrick_drive/planters/brick_001/conceptgraphs/configs/config.yaml'
CLASS_SET=ram
SCENE_NAME=conceptgraphs
THRESHOLD=1.2

python slam/cfslam_pipeline_batch.py \
    dataset_root=$ROOT_DIR \
    dataset_config=$CONFIG_PATH \
    stride=1 \
    scene_id=$SCENE_NAME \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.05 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
    dbscan_eps=0.01 \
    dbscan_min_points=1 \
    gsa_variant=ram_withbg_allclasses \
    skip_bg=False \
    max_bbox_area_ratio=10.0 \
    save_suffix=overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1 \
    obj_min_detections=0 \
    dbscan_remove_noise=False \
    downsample_voxel_size=0.01 \
    mask_area_threshold=10 \
    min_points_threshold=8 \
    semantic_threshold=0.05 \
    physical_threshold=0.05 \
    vis_render=False\
    debug_render=False \
    save_objects_all_frames=True
#    filter_interval=20