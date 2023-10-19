# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#############################################################

# Usage: python dataset_preprocessing/ffhq/preprocess_ffhq_cameras.py --source /data/ffhq --dest /data/preprocessed_ffhq_images

#############################################################

import json
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm
import argparse

COMPRESS_LEVEL = 0


def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped


def process_and_mirror_image(source, dest, filename, compress_level=COMPRESS_LEVEL):
    image_path = os.path.join(source, filename)
    img = Image.open(image_path)

    if dest is not None:  # skip saving originals if dest==source
        os.makedirs(os.path.dirname(os.path.join(dest, filename)), exist_ok=True)
        img.save(os.path.join(dest, filename), compress_level=compress_level)

    return ImageOps.mirror(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--dest", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    dest = args.source if args.dest is None else args.dest

    dataset_file = os.path.join(args.source, 'dataset.json')

    if os.path.isfile(dataset_file):  # If dataset.json present, mirror images and mirror labels.
        with open(dataset_file, "r") as f:
            dataset = json.load(f)

        max_images = args.max_images if args.max_images is not None else len(dataset['labels'])
        for i, example in tqdm(enumerate(dataset['labels']), total=max_images):
            if max_images is not None and i >= max_images:
                break
            filename, label = example
            if '_mirror' in filename:
                continue
            flipped_img = process_and_mirror_image(args.source, dest, filename)
            pose, intrinsics = np.array(label[:16]).reshape(4, 4), np.array(label[16:]).reshape(3, 3)
            flipped_pose = flip_yaw(pose)
            label = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)]).tolist()
            base, ext = filename.split('.')[0], '.' + filename.split('.')[1]
            flipped_filename = base + '_mirror' + ext
            dataset["labels"].append([flipped_filename, label])
            flipped_img.save(os.path.join(dest, flipped_filename), compress_level=COMPRESS_LEVEL)

        with open(os.path.join(dest, 'dataset.json'), "w") as f:
            json.dump(dataset, f)

    else:  # If dataset.json is not preset, just mirror images.
        for filename in tqdm(os.listdir(args.source)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                flipped_img = process_and_mirror_image(args.source, dest, filename)
                base, ext = os.path.splitext(filename)
                flipped_filename = base + '_mirror' + ext
                flipped_img.save(os.path.join(dest, flipped_filename), compress_level=COMPRESS_LEVEL)
