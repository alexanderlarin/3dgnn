import math

from .utils.rgbd_util import *


def get_hha(camera_matrix, depth, raw_depth=None):
    """
    Generate hha image from depth and raw_image (optional) images
    :param camera_matrix: Camera matrix
    :param depth: Depth image, the unit of each element in it is "meter"
    :param raw_depth: Raw depth image, the unit of each element in it is "meter"
    :return:
    """
    missing_mask = raw_depth is None
    pc, n, y_dir, h, pc_rot, n_rot = process_depth_image(depth * 100, missing_mask, camera_matrix)

    tmp = np.multiply(n, y_dir)
    acos_value = np.minimum(1, np.maximum(-1, np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acos_value.flatten()])
    angle = np.reshape(angle, h.shape)

    # must convert nan to 180 as the MATLAB program actually does.
    # or we will get a HHA image whose border region is different
    # with that of MATLAB program's output.
    angle[np.isnan(angle)] = 180

    pc[:, :, 2] = np.maximum(pc[:, :, 2], 100)
    image = np.zeros(pc.shape)

    # opencv-python save the picture in BGR order.
    image[:, :, 2] = 31000 / pc[:, :, 2]
    image[:, :, 1] = h
    image[:, :, 0] = (angle + 128 - 90)

    # print(np.isnan(angle))

    # np.uint8 seems to use 'floor', but in matlab, it seems to use 'round'.
    # So image convert it to integer myself.
    image = np.rint(image)

    # np.uint8: 256->1, but in MATLAB, uint8: 256->255
    image[image > 255] = 255
    return image.astype(np.uint8)
