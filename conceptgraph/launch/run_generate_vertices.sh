ROOT_DIR=/home/jlidard/zed2i
SCENE_NAME=$1
RESULT_PATH=$ROOT_DIR/$SCENE_NAME/pcd_saves/full_pcd_ram_withbg_allclasses_OUTSIDE_post.pkl.gz
SAVE_PATH=$ROOT_DIR/$SCENE_NAME/observation_dict.json
SAVE_PCD_RESULT_PATH=$ROOT_DIR/$SCENE_NAME/partial_pcd.ply
FULL_PCD_PATH=$ROOT_DIR/$SCENE_NAME/ptcloud/ptcloud.ply

python launch/generate_vertices.py \
--result_path $RESULT_PATH \
--json_file ${ROOT_DIR}/${SCENE_NAME}/sg_cache/captions.json \
--save_file ${SAVE_PATH} \
--downsample_ratio 0.01 \
--no_clip \
--save_partial_pcd_path ${SAVE_PCD_RESULT_PATH} \
--full_pc_path  ${FULL_PCD_PATH} 
#--skip_rendering

