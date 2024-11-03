
import os
import numpy as np
from PIL import Image

def main():
    scene_name = 'brick_001'
    root_dir = f'/home/jlidard/pbrick_drive/planters/{scene_name}/conceptgraphs'

    # load in focal length
    focal_length_path = os.path.join(root_dir, 'focal_length.npy')
    focal_length = np.load(focal_length_path)

    # get the approximate focal length based on the prediction
    fx, fy = focal_length, focal_length

    # get the image in the results subdir
    img_path = os.path.join(root_dir, 'results', f'{scene_name}.jpg')

    # load the image into memory
    img = Image.open(img_path)
    width, height = img.size

    # get the approximate image center point intrinsics based on the image dimension
    cx, cy = width / 2, height / 2

    # make a new intrinsics mat
    intrinsics = np.eye(3)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy

    # make a new intrinsics folder under root_dir
    intrinsics_subdir = os.path.join(root_dir, 'intrinsics')
    os.makedirs(intrinsics_subdir, exist_ok=True)

    # save the intrinsics mat
    np.save(os.path.join(intrinsics_subdir, 'intrinsics.npy'), intrinsics)


if __name__ == '__main__':
    main()
