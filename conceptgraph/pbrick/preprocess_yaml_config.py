import numpy as np
import os
import yaml
def main():

    yaml_file = '/home/jlidard/concept-graphs/conceptgraph/dataset/dataconfigs/zed2i/zed2i_dynamic_intrinsics.yaml'

    with open(yaml_file, "r") as file:
        cfg = yaml.safe_load(file)

    base_dir = '/home/jlidard/pbrick_drive/planters/brick_001/conceptgraphs/'
    intrinsics_file = os.path.join(base_dir, 'intrinsics/intrinsics.npy')

    intrinsics_mat = np.load(intrinsics_file)
    fx = intrinsics_mat[0, 0].item()
    fy = intrinsics_mat[1, 1].item()
    cx = intrinsics_mat[0, 2].item()
    cy = intrinsics_mat[1, 2].item()

    image_width = int(cx * 2)
    image_height = int(cy * 2)

    cfg["camera_params"]["image_width"] = image_width
    cfg["camera_params"]["image_height"] = image_height
    cfg["camera_params"]["fx"] = fx
    cfg["camera_params"]["fy"] = fy
    cfg["camera_params"]["cx"] = cx
    cfg["camera_params"]["cy"] = cy

    # make a config folder under base dir
    config_path = os.path.join(base_dir, 'configs')
    os.makedirs(config_path, exist_ok=True)

    # save a copy
    yaml.dump(cfg, open(os.path.join(config_path, 'config.yaml'), "w"))

if __name__ == '__main__':
    main()


