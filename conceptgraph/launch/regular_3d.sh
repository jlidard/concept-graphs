SCENE_NAME=241202_ECL_500_3_0.05_20_1
python scripts/run_slam_rgb.py \
    --dataset_root $ZED_ROOT \
    --dataset_config $ZED_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --image_height 720 \
    --image_width 1280 \
    --stride 1 \
    --visualize \
    --save_pcd \
#    --end 200