SCENE_NAME=lab4

# The CoceptGraphs (without open-vocab detector)
python scripts/generate_gsa_results.py \
    --dataset_root $ZED_ROOT \
    --dataset_config $ZED_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set none \
    --stride 5 \
#    --nms_threshold 0.9 \
#    --box_threshold 0.95 \
#    --text_threshold 0.9