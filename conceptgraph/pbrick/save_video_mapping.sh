SCENE_NAME=lab7
FOLDER_NAME=ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1

python scripts/animate_mapping_interactive.py --input_folder $ZED_ROOT/$SCENE_NAME/objects_all_frames/$FOLDER_NAME
python scripts/animate_mapping_save.py --input_folder $REPLICA_ROOT/$SCENE_NAME/objects_all_frames/$FOLDER_NAME