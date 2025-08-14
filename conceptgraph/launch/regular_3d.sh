SCENE_NAME=250814_ECL_200_2_0.01_20_1
# SCENE_NAME=50108_Debug_50_5_0.05_20_1
python scripts/run_slam_rgb.py \
    --dataset_root $ZED_ROOT \
    --dataset_config $ZED_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --image_height 720 \
    --image_width 1280 \
    --stride 2 \
    --visualize \
    --save_pcd \
    --start 0
#    --end 200