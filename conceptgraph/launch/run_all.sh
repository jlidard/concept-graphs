#!/bin/bash

scene_name=250416_lab_100_2_0.01_20_1

./launch/extract_2d_classes.sh ${scene_name}

./launch/run_map_classes.sh ${scene_name}

./launch/visualize_classes.sh ${scene_name}

./launch/run_scene_graph_new.sh ${scene_name}

./launch/run_generate_vertices.sh ${scene_name}