import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep
from scipy.ndimage import uniform_filter


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo (or mean albedo for RGB input images) for a pixel is less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights. All N images have the same dimension.
                  The images can either be all RGB (height x width x 3), or all grayscale (height x width x 1).

    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    height, width = images[0].shape[:2]

    # Initialize albedo and normals
    albedo = np.zeros((height, width, 3 if images[0].shape[2] == 3 else 1), dtype=np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)

    I = np.array(images, dtype=np.float32)

    # Compute pseudoinverse of lighting directions for solving the linear system
    L_PSEUDO_INV = np.linalg.inv(lights.T@lights)@lights.T
    
    # Solve for the gradient surface matrix G
    G = np.einsum('ij,j...->...i', L_PSEUDO_INV, I)
    
    albedo_values = np.sqrt(np.sum(G**2, axis=3))
    mask = albedo_values[:,:,0] >= 1e-7

    # Update albedo only for significant pixels
    albedo[mask] = albedo_values[mask]

    # Calculate and update normals for pixels with significant albedo
    normals[mask] = G[:,:,0][mask] / albedo_values[:,:,0][mask, np.newaxis]

    return albedo, normals



def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    # Homogeneous coordinates
    points_homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    matrix_projection = K @ Rt

    points_projection = np.einsum('ij,hwj->hwi', matrix_projection, points_homogeneous)

    normalized_points_projection = points_projection[..., :-1] / points_projection[..., -1, np.newaxis]
    return normalized_points_projection


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc_impl' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc_impl' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    if image.ndim == 3:
        mean = uniform_filter(image, mode='constant', cval=0, size=(ncc_size, ncc_size, 1),  output=np.float64,)
    elif image.ndim == 2:
        mean = uniform_filter(image, size=ncc_size, mode='constant', cval=0)
    
    height, width, channels = image.shape
    normalized = np.zeros((height, width, channels * ncc_size**2), dtype=np.float32)

    window_size = ncc_size//2
    
    for w in range(window_size, width - window_size):
        for h in range(window_size, height - window_size):
            patch = image[h - window_size:h + window_size + 1, w - window_size:w + window_size + 1] - mean[h, w]
            patch_vector = patch.reshape(ncc_size**2, -1).flatten(order='F')

            norm = np.linalg.norm(patch_vector)
            if norm < 1e-6:
                normalized[h, w] = 0
            else:
                normalized[h, w, :] = patch_vector / norm

    return normalized 


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc_impl.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    # Compute dot product between vectors
    ncc = np.sum(image1 * image2, axis=2)
    return ncc
