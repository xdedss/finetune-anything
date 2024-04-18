
import json
import os
import sys
import time
import random

import fire
import tqdm

import numpy as np
import cv2
from PIL import Image


# 0 no building
# 1 no damage
# 2-4 different damage levels
building_desc = [
    [],
    ['buildings in good shape'],
    ['buildings with slight damage'],
    ['buildings that are severely damaged'],
    ['buildings that are completely destroyed'],
]

def build_subset(subset_root, output_root):
    images_dir = os.path.join(subset_root, 'images')
    targets_dir = os.path.join(subset_root, 'targets')

    images_list = os.listdir(images_dir)
    images_list.sort()
    
    # remove pre disaster since we are generating post dataset with pre prompts
    images_list = [path for path in images_list if '_post_' in path]

    images_ids = [path.replace('.png', '') for path in images_list]
    targets_list_expected = [f'{id}_target.png' for id in images_ids]

    for target_fname in targets_list_expected:
        assert os.path.isfile(os.path.join(targets_dir, target_fname))
    
    # [{"image": "path/to", "mask": "path/to", "prompt": "xxx"}, ...]
    res = []
    for image_id, image_fname, target_fname in zip(tqdm.tqdm(images_ids), images_list, targets_list_expected):
        # print(image_fname, target_fname)
        # analyze label
        image_path = os.path.join(images_dir, image_fname)
        target_path = os.path.join(targets_dir, target_fname)
        target_path_pre = target_path.replace('_post_', '_pre_')

        image = Image.open(image_path).convert('RGB')
        target = Image.open(target_path)
        target_np = np.array(target)
        
        target_pre = Image.open(target_path_pre)
        target_pre_np = np.array(target_pre)
        
        mask_pre = target_pre_np > 0
        output_prompt_mask_path = os.path.join(output_root, 'targets', f'{image_id.replace("_post_", "_pre_")}.png')
        cv2.imwrite(output_prompt_mask_path, mask_pre.astype(np.uint8) * 255)

        unique_labels = set(np.unique(target_np))

        # different levels
        for label_id in unique_labels:
            # skip bg for now
            if (label_id == 0):
                continue
            mask = target_np == label_id
            prompt = random.choice(building_desc[label_id])


            # wirte to disk
            output_image_path = os.path.join(output_root, 'images', f'{image_id}.png') # same image for different label_ids
            output_mask_path = os.path.join(output_root, 'targets', f'{image_id}_{label_id}.png')

            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
            cv2.imwrite(output_mask_path, mask.astype(np.uint8) * 255)
            image.save(output_image_path)
            res.append({
                "image": os.path.abspath(output_image_path),
                "mask": os.path.abspath(output_mask_path),
                "prompt": prompt,
                "prompt_mask": os.path.abspath(output_prompt_mask_path),
            })
        
        

        # print(unique_labels)
        # print(target_np)

        # sys.exit(0)
            
    with open(os.path.join(output_root, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f)


def main(xbd_root):
    '''
    # v1: build post disaster only

    # 0 no building
    # 1 no damage
    # 2-4 different damage levels

    - test/
        - images/
            socal-fire_00001400_post_disaster.png
            socal-fire_00001400_pre_disaster.png
            ...
        - targets/
            socal-fire_00001400_post_disaster_target.png
            socal-fire_00001400_pre_disaster_target.png
            ...
    - train/
        - images/
            ...
        - targets/
            ...

    '''

    test_root = os.path.join(xbd_root, 'test')
    train_root = os.path.join(xbd_root, 'train')

    build_subset(test_root, 'dataset_scripts/built/xbd_pre_prompt/test')
    build_subset(train_root, 'dataset_scripts/built/xbd_pre_prompt/train')




if __name__ == '__main__':
    fire.Fire(main)

