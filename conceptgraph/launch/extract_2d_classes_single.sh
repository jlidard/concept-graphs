# The ConceptGraphs-Detect
SCENE_ID=$1
ROOT_DIR=/home/pbrick/dev/data # /$SCENE_ID
# CONFIG_PATH=/home/pbrick/dev/data/$SCENE_ID/conceptgraphs/configs/config.yaml
# CONFIG_PATH='/home/jlidard/concept-graphs/conceptgraph/dataset/dataconfigs/zed2i/zed2i_dynamic_intrinsics.yaml'
CONFIG_PATH='/home/pbrick/concept-graphs/conceptgraph/dataset/dataconfigs/zed2i/zed2i.yaml'
CLASS_SET=ram
SCENE_NAME=250814_ECL_200_2_0.01_20_1

python scripts/generate_gsa_results.py \
    --dataset_root $ROOT_DIR \
    --dataset_config $CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set $CLASS_SET \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --stride 1 \
    --add_bg_classes \
    --exp_suffix withbg_allclasses