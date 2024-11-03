
import os
import numpy as np
from PIL import Image

def main():
    scene_name = 'brick_001'
    root_dir = f'/home/jlidard/pbrick_drive/planters/{scene_name}/conceptgraphs'

    # example pose
    example_pose_path = '/home/jlidard/zed2i/ecl_4/poses/frame000001.npy'
    pose = np.load(example_pose_path)

    pose_dir = os.path.join(root_dir, 'poses')
    os.makedirs(pose_dir, exist_ok=True)

    np.save(os.path.join(pose_dir, 'pose.npy'), pose)


if __name__ == '__main__':
    main()
