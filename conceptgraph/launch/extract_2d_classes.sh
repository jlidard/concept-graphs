# The ConceptGraphs-Detect
CLASS_SET=ram
SCENE_NAME=241127_ECL_500_5_0.05_20_1

python scripts/generate_gsa_results.py \
    --dataset_root $ZED_ROOT \
    --dataset_config $ZED_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set $CLASS_SET \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --stride 50 \
    --add_bg_classes \
    --accumu_classes \
    --exp_suffix withbg_allclasses