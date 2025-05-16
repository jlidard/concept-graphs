SCENE_NAME=250415_lab_100_2_0.001_20_1
python scripts/run_slam_rgb.py \
    --dataset_root $ZED_ROOT \
    --dataset_config $ZED_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --image_height 1242 \
    --image_width 2208 \
    --stride 5 \
    --visualize \
    --save_pcd \
#    --end 200