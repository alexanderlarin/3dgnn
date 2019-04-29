import cv2
import h5py
import numpy as np
import logging
import os
from itertools import repeat
from multiprocessing import Pool

from hha import get_hha

logger = logging.getLogger()


def get_camera_matrix():
    # From http://cs.nyu.edu/~silberman/code/toolbox_nyu_depth_v2.zip camera_params.m
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02
    return np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])


def export_hha(depth_image, hha_filename, color_camera_matrix):
    hha = get_hha(color_camera_matrix, depth_image.T, depth_image.T)
    cv2.imwrite(hha_filename, hha)


def extract_hha(dataset_filename, hha_dir, color_camera_matrix=None, mp_workers=4, mp_chunk_size=16):
    if not os.path.exists(dataset_filename):
        logger.error(f'Labeled dataset {dataset_filename} not found')
    else:
        if color_camera_matrix is None:
            color_camera_matrix = get_camera_matrix()
        with h5py.File(dataset_filename) as data_file:
            logger.info(f'Labeled dataset {dataset_filename} loaded')
            depth_count = data_file['depths'].shape[0]
            logger.info(f'Depth images count={depth_count}')
            if not os.path.exists(hha_dir) or not os.path.isdir(hha_dir):
                os.mkdir(hha_dir)
                logger.warning(f'HHA dir is not exists, created dir="{hha_dir}"')
            start_idx = 0
            for _, _, filenames in os.walk(hha_dir):
                for filename in sorted(filenames):
                    name, ext = os.path.splitext(filename)
                    start_idx = max(start_idx, int(name))
            if start_idx != 0:
                logger.warning(f'HHA files dir is not empty, continue from [{start_idx + 1}]')
            with Pool(mp_workers) as pool:
                for start_chunk_idx in range(start_idx, depth_count, mp_chunk_size):
                    end_chunk_idx = start_chunk_idx + mp_chunk_size - 1
                    depth_images = data_file['depths'][start_chunk_idx:end_chunk_idx, :]
                    hha_filenames = (os.path.join(hha_dir, f'{idx + 1}.png')
                                     for idx in range(start_chunk_idx, end_chunk_idx))
                    pool.starmap(export_hha, zip(depth_images, hha_filenames,
                                                 repeat(color_camera_matrix, end_chunk_idx - start_chunk_idx)))
                    logger.info(f'HHA converted [{end_chunk_idx + 1}/{depth_count}]')

        logger.info(f'HHA converting successfully completed')
