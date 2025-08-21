#!/bin/bash

# TODO: set env variable for conceptgraph and/or make it a submodule
scene_name=250820_ECL_500_2_0.01_20_1

echo "extract_2d_classes"
./launch/extract_2d_classes.sh ${scene_name}

echo "run_map_classes"
./launch/run_map_classes.sh ${scene_name}

echo "visualize_classes"
./launch/visualize_classes.sh ${scene_name}

echo "run_scene_graph_new"
./launch/run_scene_graph_new.sh ${scene_name}

echo "run_generate_vertices"
./launch/run_generate_vertices.sh ${scene_name}