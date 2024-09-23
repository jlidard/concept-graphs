SCENE_NAME=lab7
python scripts/run_slam_rgb.py \
    --dataset_root $ZED_ROOT \
    --dataset_config $ZED_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --image_height 720 \
    --image_width 1280 \
    --stride 5 \
    --visualize \
#    --end 100