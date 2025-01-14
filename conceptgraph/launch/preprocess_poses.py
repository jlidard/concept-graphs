
import os
import numpy as np
from PIL import Image

def main(root_dir, scene_name, annotation_dir):
    # example pose
    example_pose_path = '/home/jlidard/zed2i/ecl_4/poses/frame000001.npy'
    pose = np.load(example_pose_path)


    pose_path = os.path.join(root_dir, annotation_dir, 'inverse_custom_axes_transform.npy')
    pose_annotated = np.load(pose_path)

    pose_dir = os.path.join(root_dir, 'conceptgraphs', 'poses')

    if annotation_dir == "depth_pro":
        os.makedirs(pose_dir, exist_ok=True)
        np.save(os.path.join(pose_dir, 'pose.npy'), pose_annotated)
    else:
        new_pose_dir = os.path.join(root_dir, 'conceptgraphs', 'poses_new')
        os.makedirs(new_pose_dir, exist_ok=True)
        np.save(os.path.join(pose_dir, 'pose.npy'), pose_annotated)
        all_poses = os.listdir(pose_dir)
        all_poses = [p for p in all_poses if p.endswith('.npy')]
        for pose in all_poses:
            existing_pose = np.load(os.path.join(pose_dir, pose))
            transformed_pose = pose_annotated @ existing_pose
            np.save(os.path.join(new_pose_dir, pose), transformed_pose)

def preprocess_icl(annotation_dir='depth_pro'):

    planters_dir = '/home/jlidard/pbrick_drive/planters'
    all_scenes = os.listdir(planters_dir)
    all_scenes.sort()
    all_scenes = all_scenes[1:]  # remove .DS_Store
    for scene_name in all_scenes:
        try:
            root_dir = f'/home/jlidard/pbrick_drive/planters/{scene_name}/'
            main(root_dir, scene_name, annotation_dir)
        except Exception as e:
            print(e)
            print('Moving on the scene: {}'.format(scene_name))

def preprocess_ecl(annotation_dir='pose_annotation'):

    planters_dir = '/home/jlidard/pbrick_drive/ECL_buildable'
    all_scenes = os.listdir(planters_dir)
    all_scenes.sort()
    all_scenes = all_scenes # remove .DS_Store
    for scene_name in all_scenes:
        try:
            root_dir = f'{planters_dir}/{scene_name}/'
            main(root_dir, scene_name, annotation_dir)
        except Exception as e:
            print(e)
            print('Moving on the scene: {}'.format(scene_name))


if __name__ == '__main__':
    preprocess_ecl()
