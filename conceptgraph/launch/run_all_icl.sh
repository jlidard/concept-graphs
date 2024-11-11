# runs conceptgraph on entire ICL dataset

# preprocess the dataset to extract intrinsics, poses, and camera config
#python launch/preprocess_intrinsics.py
#python launch/preprocess_poses.py
#python launch/preprocess_yaml_config.py

# extract 2d features
path="/home/jlidard/pbrick_drive/planters"
subdirs=($(find "$path" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort))
echo "$subdirs"

#script_path="./launch/extract_2d_classes_single.sh"
#start_num=16
#stop_num=26
#echo ${#subdirs[@]}
#
#for i in $(seq $start_num $stop_num); do
#  if (( i - 1 < ${#subdirs[@]} )); then
#    scene_path="${subdirs[$((i - 1))]}"  # Adjust for zero-based indexing
#    echo "Processing scene $scene_path"
#  else
#    echo "Index $i is out of bounds for the array 'subdirs'."
#    continue
#  fi
#  bash "$script_path" "$scene_path"
#done


#script_path="./launch/run_map_classes_single.sh"
#start_num=6
#stop_num=26
#echo ${#subdirs[@]}
#
#for i in $(seq $start_num $stop_num); do
#  if (( i - 1 < ${#subdirs[@]} )); then
#    scene_path="${subdirs[$((i - 1))]}"  # Adjust for zero-based indexing
#    echo "Processing scene $scene_path"
#  else
#    echo "Index $i is out of bounds for the array 'subdirs'."
#    continue
#  fi
#  bash "$script_path" "$scene_path"
#done


script_path="./launch/visualize_classes_single.sh"
start_num=1
stop_num=25
echo ${#subdirs[@]}

for i in $(seq $start_num $stop_num); do
  if (( i - 1 < ${#subdirs[@]} )); then
    scene_path="${subdirs[$((i - 1))]}"  # Adjust for zero-based indexing
    echo "Processing scene $scene_path"
  else
    echo "Index $i is out of bounds for the array 'subdirs'."
    continue
  fi
  bash "$script_path" "$scene_path"
done


#  bash run_map_classes_single.sh $scene_path
#  bash run_scene_graph $scene_path $path_tbd
