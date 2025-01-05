from scipy.ndimage import affine_transform
import numpy as np


def get_random_transform_params(image_size):
    theta = np.random.random() * 2*np.pi
    trans = np.random.randint(0, image_size//10, 2) - image_size//20
    pivot = (image_size / 2, image_size / 2)
    return theta, trans, pivot


def get_image_transform(theta, trans, pivot=(0, 0)):
    """Compute composite 2D rigid transformation matrix."""
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_t_image = np.array([[1., 0., -pivot[0]], [0., 1., -pivot[1]],
                              [0., 0., 1.]])
    image_t_pivot = np.array([[1., 0., pivot[0]], [0., 1., pivot[1]],
                              [0., 0., 1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]], [0., 0., 1.]])
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))


def perturb(obs, next_obs, state, next_state, dxy,
            theta, trans, pivot,
            set_theta_zero=False, set_trans_zero=False):
    """Perturn an image for data augmentation"""

    if set_theta_zero:
        theta = 0.
    if set_trans_zero:
        trans = [0., 0.]
    transform = get_image_transform(theta, trans, pivot)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    rotated_dxy = rot.dot(dxy)
    rotated_dxy = np.clip(rotated_dxy, -1, 1)

    # Apply rigid transform to image and pixel labels.
    obs[0] = affine_transform(obs[0], np.linalg.inv(transform),
                              mode='nearest', order=1)
    obs[1] = affine_transform(obs[1], np.linalg.inv(transform),
                              mode='nearest', order=1)
    state[0] = affine_transform(state[0], np.linalg.inv(transform),
                                mode='nearest', order=1)
    state[1] = affine_transform(state[1], np.linalg.inv(transform),
                                mode='nearest', order=1)
    next_obs[0] = affine_transform(next_obs[0], np.linalg.inv(transform),
                                   mode='nearest', order=1)
    next_obs[1] = affine_transform(next_obs[1], np.linalg.inv(transform),
                                   mode='nearest', order=1)
    next_state[0] = affine_transform(next_state[0], np.linalg.inv(transform),
                                     mode='nearest', order=1)
    next_state[1] = affine_transform(next_state[1], np.linalg.inv(transform),
                                     mode='nearest', order=1)
    return obs, next_obs, state, next_state, rotated_dxy

def perturb_1(obs, state, dxy, theta, trans, pivot,
            set_theta_zero=False, set_trans_zero=False):
    """Perturn an image for data augmentation"""

    if set_theta_zero:
        theta = 0.
    if set_trans_zero:
        trans = [0., 0.]
    transform = get_image_transform(theta, trans, pivot)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    rotated_dxy = rot.dot(dxy)
    rotated_dxy = np.clip(rotated_dxy, -1, 1)

    # Apply rigid transform to image and pixel labels.
    obs[0] = affine_transform(obs[0], np.linalg.inv(transform),
                              mode='nearest', order=1)
    obs[1] = affine_transform(obs[1], np.linalg.inv(transform),
                              mode='nearest', order=1)
    state[0] = affine_transform(state[0], np.linalg.inv(transform),
                                mode='nearest', order=1)
    state[1] = affine_transform(state[1], np.linalg.inv(transform),
                                mode='nearest', order=1)
    return obs, state, rotated_dxy