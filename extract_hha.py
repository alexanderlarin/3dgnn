import multiprocessing

import cv2
import h5py
import numpy as np
import logging
from itertools import repeat
from multiprocessing import Pool
from tqdm import tqdm

from hha import get_hha

logger = logging.getLogger()


CHANNELS_COUNT = 3


def get_camera_matrix():
    # From http://cs.nyu.edu/~silberman/code/toolbox_nyu_depth_v2.zip camera_params.m
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02
    return np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])


def get_hha_rgb(depth_image, color_camera_matrix):
    hha_bgr = get_hha(color_camera_matrix, depth_image.T, depth_image.T)
    return cv2.cvtColor(hha_bgr, cv2.COLOR_BGR2RGB)


def patch_hha_dataset(dataset_filename, patch_dataset_filename,
                      color_camera_matrix=None, mp_workers=multiprocessing.cpu_count(), mp_chunk_size=16):
    with h5py.File(dataset_filename, mode='r') as dataset_file, h5py.File(patch_dataset_filename) as patch_dataset_file:
        depths = dataset_file['depths']
        depths_count, width, height = depths.shape
        logger.info(f'Labeled dataset {dataset_filename} loaded')
        logger.info(f'Depth images count={depths_count}')
        logger.info(f'Workers: {mp_workers}, Chunk: {mp_chunk_size}')

        try:
            hha_images = patch_dataset_file.require_dataset('hha_images',
                                                            shape=(0, CHANNELS_COUNT, width, height),
                                                            maxshape=(depths_count, CHANNELS_COUNT, width, height),
                                                            dtype=np.uint8, chunks=True)
        except TypeError:
            hha_images = patch_dataset_file['hha_images']

        start_idx = hha_images.shape[0]
        logger.warning(f'{patch_dataset_filename} dataset has pre-generated HHA images, continue from {start_idx}')

        if color_camera_matrix is None:
            color_camera_matrix = get_camera_matrix()
            logger.warning(f'no camera matrix passed, use DEFAULT:\n{color_camera_matrix}')

        with Pool(mp_workers) as pool, tqdm(desc='HHA', initial=start_idx, total=depths_count) as progress_bar:
            for start_chunk_idx in range(start_idx, depths_count, mp_chunk_size):
                depth_images = depths[start_chunk_idx:start_chunk_idx + mp_chunk_size, :]
                hha_images_chunk = pool.starmap(get_hha_rgb, zip(depth_images,
                                                                 repeat(color_camera_matrix, mp_chunk_size)))
                hha_images.resize(hha_images.shape[0] + len(hha_images_chunk), axis=0)
                for offset, hha_image in enumerate(hha_images_chunk):
                    hha_images[start_chunk_idx + offset] = np.transpose(hha_image, [2, 1, 0])
                patch_dataset_file.flush()
                progress_bar.update(mp_chunk_size)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    patch_hha_dataset('datasets/data/nyu_depth_v2_labeled.mat',
                      'datasets/data/nyu_depth_v2_patch.mat',
                      mp_workers=2, mp_chunk_size=4)
