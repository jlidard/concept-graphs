SCENE_NAME=241014_ECL_500_5_0.05_20_1
python scripts/run_slam_rgb.py \
    --dataset_root $ZED_ROOT \
    --dataset_config $ZED_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --image_height 1242 \
    --image_width 2208 \
    --stride 75 \
    --visualize \
    # --end 1000
