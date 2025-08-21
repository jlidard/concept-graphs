import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gpt4o_interface import GPT4OInterface


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)

    return parser


# Function to extract the numeric part
def extract_number(filename):
    # Split the filename to extract the number
    return int(os.path.splitext(filename.split("/")[-1])[0])


def plot_images_with_captions(images, captions, savedir, idx_obj):
    """This is debug helper function that plots the images with the captions and masks overlaid and saves them to a directory. This way you can inspect exactly what the LLaVA model is captioning which image with the mask, and the mask confidence scores overlaid."""

    n = min(9, len(images))  # Only plot up to 9 images
    nrows = int(np.ceil(n / 3))
    ncols = 3 if n > 1 else 1
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)  # Adjusted figsize

    for i in range(n):
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        ax.imshow(images[i])

        # Apply the mask to the image
        img_array = np.array(images[i])
        # if img_array.shape[:2] != masks[i].shape:
        #     ax.text(0.5, 0.5, "Plotting error: Shape mismatch between image and mask", ha='center', va='center')
        # else:
        #     green_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
        #     green_mask[masks[i]] = [0, 255, 0]  # Green color where mask is True
        #     ax.imshow(green_mask, alpha=0.15)  # Overlay with transparency

        title_text = f"Label: {captions[i]}"

        ax.set_title(title_text, fontsize=12)  # Reduced font size for better fitting
        ax.axis("off")

    # Remove any unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f"{idx_obj}.png"))
    plt.close()


def extract_node_captions(args):
    root_dir = args.root_dir
    image_dir = args.image_dir

    # get all images in image_dir
    image_dir_path = str(os.path.join(root_dir, image_dir))
    all_images = os.listdir(image_dir_path)
    all_images_path = [os.path.join(image_dir_path, p) for p in all_images]
    # remove any non-png images
    all_images_path = [p for p in all_images_path if p.endswith(".png")]
    # sort numerically
    all_images_path = sorted(all_images_path, key=extract_number)

    model_str = "gpt-4o"

    system_prompt = (
        "You are an expert on classifiying objects in images. Follow the instructions exactly, and answer concisely."
    )
    user_prompt = (
        "Describe the main object enclosed by the red outline in 1-2 words. If multiple "
        "images are provided, provide the best 1-2 word summary for all images. "
        "If you're unsure or the whole object isn't in the image, say 'ignore'."
    )
    model = GPT4OInterface(model=model_str, system_prompt=system_prompt, max_tokens=200, return_logprobs=True)  # mcqa

    captions = []
    for image_path in all_images_path:
        image_paths_input = [image_path]
        result, _, _ = model.query(user_prompt, image_path=image_paths_input)
        captions.append(result)
        print(image_path)
        print(result)
        print()

    # save the results
    save_dir_captioned_images = str(os.path.join(root_dir, "captioned_images"))
    os.makedirs(save_dir_captioned_images, exist_ok=True)
    for i, image_path in enumerate(all_images_path):
        caption = captions[i]
        image = Image.open(image_path)
        plot_images_with_captions([image], captions=[caption], savedir=save_dir_captioned_images, idx_obj=i)

    # save the captions
    with open(os.path.join(root_dir, "captions.json"), "w") as f:
        json.dump(captions, f, indent=4)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    extract_node_captions(args)