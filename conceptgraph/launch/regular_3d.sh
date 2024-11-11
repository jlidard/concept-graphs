SCENE_NAME=gen2_test_5
python scripts/run_slam_rgb.py \
    --dataset_root $ZED_ROOT \
    --dataset_config $ZED_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --image_height 720 \
    --image_width 1280 \
    --stride 10 \
    --visualize \
    --save_pcd \
#    --end 200