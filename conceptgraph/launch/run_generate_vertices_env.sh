SCENE_ID=$1
ROOT_DIR=/home/jlidard/pbrick_drive/ECL_buildable/$SCENE_ID
SCENE_NAME=conceptgraphs
RESULT_PATH=$ROOT_DIR/$SCENE_NAME/pcd_saves/full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz
SAVE_PATH=$ROOT_DIR/$SCENE_NAME/observation_dict.pkl
SAVE_PCD_RESULT_PATH=$ROOT_DIR/$SCENE_NAME/partial_pcd.ply

python launch/generate_vertices.py \
--result_path $RESULT_PATH \
--json_file ${ROOT_DIR}/${SCENE_NAME}/sg_cache/captions.json \
--save_file ${SAVE_PATH} \
--downsample_ratio 0.01 \
--no_clip \
--save_partial_pcd_path ${SAVE_PCD_RESULT_PATH}
#--skip_rendering

