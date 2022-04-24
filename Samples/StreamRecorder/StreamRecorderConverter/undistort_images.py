"""
 Copyright (c) Microsoft. All rights reserved.
 This code is licensed under the MIT License (MIT).
 THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
 ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
 IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
 PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
"""
from genericpath import exists
import os
import cv2
import argparse
import numpy as np
import multiprocessing
from pathlib import Path
import scipy.interpolate

from utils import folders_extensions, load_lut


def undistort_image(input_folder, output_folder, image_name, lut, undist_size):
    image_path = input_folder / image_name
    output_path = output_folder / image_name
    print(f"Read image {image_path}")
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    except BaseException as err:
        print(f"Unexpected {err=}, {type(err)=}")

    w, h = img.shape

    assert(lut.shape[0] == w * h)

    xx, yy = np.meshgrid(np.arange(0, undist_size[0]), np.arange(0, undist_size[1]))

    undist_img = np.clip(np.reshape(scipy.interpolate.griddata((lut[:,0], lut[:,1]),np.reshape(img, (-1,1), order='C'),(xx, yy),method='linear', fill_value=255), (undist_size[1], undist_size[0]), order='C'), 0, 255)

    cv2.imwrite(str(output_path), undist_img)


def undistort_images(folder):
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    for (img_folder, extension) in folders_extensions:
        # if img_folder[:3] == 'VLC':
        if img_folder == 'VLC LL':
            print(f"Undistorting images in {img_folder}")
            # pv_path = list(folder.glob('*pv.txt'))
            lut_path = folder / (img_folder + '_lut.bin')
            lut = load_lut(lut_path)

            undist_size = (640, 480)
            undist_K = np.array([
                [400, 0, undist_size[0]/2],
                [0, 400, undist_size[1]/2],
                [0, 0, 1]
            ])

            # We adapt the lookup table to directly map to coordinates in the undistorted image
            im_lut = lut / np.tile(lut[:,[2]], [1,3])
            im_lut = np.transpose(undist_K @ np.transpose(im_lut))

            undist_path = folder / img_folder / 'undist'
            undist_path.mkdir(exist_ok=True)

            np.savetxt(undist_path / 'K.txt', undist_K)
            np.savetxt(undist_path / 'image_size.txt', undist_size)

            input_folder = folder / img_folder
            img_names = [p.relative_to(input_folder) for p in input_folder.glob('*pgm')]

            print(f"Processing images in {img_folder}")
            for img_name in img_names:
                p.apply_async(undistort_image, (input_folder, undist_path, img_name, im_lut, undist_size))
    p.close()
    p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Undistort images')
    parser.add_argument("--recording_path", required=True,
                        help="Path to recording folder")
    args = parser.parse_args()
    undistort_images(Path(args.recording_path))
