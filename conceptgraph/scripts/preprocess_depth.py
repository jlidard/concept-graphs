from glob import glob
import os
import numpy as np

results_dir = "/home/pbrick/dev/data/250820_ECL_500_2_0.01_20_1/results"

depth_paths = sorted(glob(os.path.join(results_dir, "depth*.npy")))
conf_paths = sorted(glob(os.path.join(results_dir, "conf*.npy")))

min_depth = 0.3
max_depth = 2
min_conf = 20

for depth_path, conf_path in zip(depth_paths, conf_paths):
    base_name = os.path.basename(depth_path)
    depth = np.load(depth_path)
    conf = np.load(conf_path)
    depth[depth < min_depth] = np.nan
    depth[depth > max_depth] = np.nan
    depth[conf > min_conf] = np.nan
    np.save(os.path.join(results_dir, f"filt_{base_name}"), depth)
