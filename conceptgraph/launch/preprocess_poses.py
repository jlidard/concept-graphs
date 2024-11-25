
import os
import numpy as np
from PIL import Image

def main(root_dir, scene_name):
    # example pose
    example_pose_path = '/home/jlidard/zed2i/ecl_4/poses/frame000001.npy'
    pose = np.load(example_pose_path)

    pose_path = os.path.join(root_dir, 'depth_pro', 'inverse_custom_axes_transform.npy')
    pose_annotated = np.load(pose_path)

    pose_dir = os.path.join(root_dir, 'conceptgraphs', 'poses')
    os.makedirs(pose_dir, exist_ok=True)

    np.save(os.path.join(pose_dir, 'pose.npy'), pose_annotated)


if __name__ == '__main__':
    planters_dir = '/home/jlidard/pbrick_drive/planters'
    all_scenes = os.listdir(planters_dir)
    all_scenes.sort()
    all_scenes = all_scenes[1:]  # remove .DS_Store
    for scene_name in all_scenes:
        try:
            root_dir = f'/home/jlidard/pbrick_drive/planters/{scene_name}/'
            main(root_dir, scene_name)
        except Exception as e:
            print(e)
            print('Moving on the scene: {}'.format(scene_name))
