
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


# Rescuenet Defs
# https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation
building_desc = [
    ['Background'],
    ['Water'],
    ['Building No Damage'],
    ['Building Minor Damage'],
    ['Building Major Damage'],
    ['Building Total Destruction'],
    ['Road-Clear'],
    ['Road-Blocked'],
    ['Vehicle'],
    ['Tree'],
    ['Pool']
]

building_desc_hierarchy = [
    (['Buildings'], [2, 3, 4, 5]),
    (['Buildings_with_damage'], [2, 3, 4, 5]),
]

def build_subset(subset_root, output_root):
    images_dir = os.path.join(subset_root, 'org-img')
    targets_dir = os.path.join(subset_root, 'label-img')

    images_list = os.listdir(images_dir)
    images_list.sort()
    images_ids = [path.replace('.jpg', '') for path in images_list]
    targets_list_expected = [f'{id}_lab.png' for id in images_ids]

    for target_fname in targets_list_expected:
        assert os.path.isfile(os.path.join(targets_dir, target_fname))
    
    # [{"image": "path/to", "mask": "path/to", "prompt": "xxx"}, ...]
    res = []
    for image_id, image_fname, target_fname in zip(tqdm.tqdm(images_ids), images_list, targets_list_expected):
        # print(image_fname, target_fname)
        # analyze label
        image_path = os.path.join(images_dir, image_fname)
        target_path = os.path.join(targets_dir, target_fname)

        image = Image.open(image_path).convert('RGB')
        target = Image.open(target_path)
        target_np = np.array(target)

        unique_labels = set(np.unique(target_np))

        # different levels
        for label_id in unique_labels:
            # skip bg for now
            if (label_id == 0):
                continue
            mask = target_np == label_id
            prompt = random.choice(building_desc[label_id])

            # wirte to disk
            output_image_path = os.path.join(output_root, 'images', f'{image_id}.jpg') # same image for different label_ids
            output_mask_path = os.path.join(output_root, 'targets', f'{image_id}_{label_id}.png')
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
            cv2.imwrite(output_mask_path, mask.astype(np.uint8) * 255)
            if not os.path.exists(output_image_path):
                image.save(output_image_path)
            res.append({
                "image": os.path.abspath(output_image_path),
                "mask": os.path.abspath(output_mask_path),
                "prompt": prompt,
            })
        
        # altogether
        # 可以做多层次标签
        # mask = target_np > 0
        # prompt = 'buildings'
        # output_image_path = os.path.join(output_root, 'images', f'{image_id}.png') # same image for different label_ids
        # output_mask_path = os.path.join(output_root, 'targets', f'{image_id}_all.png')
        # os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        # os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        # cv2.imwrite(output_mask_path, mask.astype(np.uint8) * 255)
        # image.save(output_image_path)
        # res.append({
        #     "image": os.path.abspath(output_image_path),
        #     "mask": os.path.abspath(output_mask_path),
        #     "prompt": prompt,
        # })
        

        # print(unique_labels)
        # print(target_np)

        # sys.exit(0)
            
    with open(os.path.join(output_root, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f)


def main(rescuenet_root):
    '''
    # Rescuenet

    - test/
        - org-img/
            12345.jpg
            ...
        - label-img/
            12345_lab.png
            ...
    - train/
        - org-img/
            ...
        - label-img/
            ...

    '''

    val_root = os.path.join(rescuenet_root, 'val')
    test_root = os.path.join(rescuenet_root, 'test')
    train_root = os.path.join(rescuenet_root, 'train')

    build_subset(val_root, 'dataset_scripts/built/rescuenet_v2/val')
    build_subset(test_root, 'dataset_scripts/built/rescuenet_v2/test')
    build_subset(train_root, 'dataset_scripts/built/rescuenet_v2/train')




if __name__ == '__main__':
    fire.Fire(main)

