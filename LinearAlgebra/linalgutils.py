# this file contains helper functions to convert between different representations

import numpy as np


def normalize_quaternion(quat):
    """
    Normalizes a quaternion.

    This function takes a quaternion as input and returns a normalized quaternion with the same orientation but with a magnitude (length) equal to 1.

    Args:
    quat (tuple): A tuple containing the quaternion (w, x, y, z).

    Returns:
    normalized_quat (tuple): A normalized quaternion (w, x, y, z) with the same orientation as the input quaternion and a magnitude (length) of 1.
    """
    norm = np.linalg.norm(quat)
    return tuple(q / norm for q in quat)


def quaternion_to_se3(quat):
    """
    Converts a quaternion to a special Euclidean group (SE(3)) matrix.

    This function takes a quaternion as input and returns a 4x4 special Euclidean group (SE(3)) matrix representing the same rotation. The translation component of the SE(3) matrix is set to zero.

    Args:
    quat (numpy.ndarray): A 1D array containing the quaternion (w, x, y, z).

    Returns:
    se3 (numpy.ndarray): A 4x4 special Euclidean group (SE(3)) matrix representing the rotation described by the input quaternion.

    Raises:
    ValueError: If the input quaternion has zero length.
    """
    w, x, y, z = quat
    Nq = w*w + x*x + y*y + z*z
    if Nq < np.finfo(float).eps:
        raise ValueError("Input quaternion has zero length.")

    s = 2.0 / Nq
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs

    se3 = np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy, 0],
        [xy + wz, 1.0 - (xx + zz), yz - wx, 0],
        [xz - wy, yz + wx, 1.0 - (xx + yy), 0],
        [0, 0, 0, 1]
    ])

    return se3


def so3_to_quaternion(R):
    """
    Brief: Convert a rotation matrix R in SO(3) to a quaternion representation.

    This function takes an input matrix R representing a 3D rotation (SO(3))
    and converts it to a quaternion representation using the algorithm outlined
    in the following paper: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    Args:
        R (numpy.ndarray): A 3x3 matrix representing a 3D rotation (SO(3)).

    Returns:
        numpy.ndarray: A 1x4 numpy array representing the quaternion (qw, qx, qy, qz).
    """
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


def quaternion_to_so3(q):
    """
    Converts a quaternion to the special orthogonal group SO(3).

    Parameters:
    q (list or np.ndarray): The quaternion as a list or numpy array [w, x, y, z],
                            where w is the scalar component and x, y, z are the vector components.

    Returns:
    np.ndarray: The 3x3 rotation matrix representing the equivalent SO(3) transformation.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Normalize quaternion
    norm = np.linalg.norm(q)
    if norm != 0:
        q = q / norm

    # Calculate rotation matrix elements
    wx, wy, wz = w * x, w * y, w * z
    xx, xy, xz = x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z

    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * (yy + zz)
    R[0, 1] = 2 * (xy - wz)
    R[0, 2] = 2 * (xz + wy)
    R[1, 0] = 2 * (xy + wz)
    R[1, 1] = 1 - 2 * (xx + zz)
    R[1, 2] = 2 * (yz - wx)
    R[2, 0] = 2 * (xz - wy)
    R[2, 1] = 2 * (yz + wx)
    R[2, 2] = 1 - 2 * (xx + yy)

    return R


def orthonormalize_matrix(M):
    """
    Brief: Orthonormalizes a given SO(3) 3x3 matrix using the Gram-Schmidt process.

    This function takes an input matrix M and applies the Gram-Schmidt process to
    orthogonalize and normalize its column vectors, resulting in an orthonormal matrix.
    The input matrix M should represent a 3D rotation matrix (SO(3)).

    Args:
        M (numpy.ndarray): A 3x3 matrix representing a 3D rotation (SO(3)).

    Returns:
        numpy.ndarray: The orthonormalized 3x3 matrix.
    """

    u1 = M[:, 0]
    u2 = M[:, 1] - np.dot(M[:, 1], u1) * u1
    u3 = M[:, 2] - np.dot(M[:, 2], u1) * u1 - np.dot(M[:, 2], u2) * u2

    u1 /= np.linalg.norm(u1)
    u2 /= np.linalg.norm(u2)
    u3 /= np.linalg.norm(u3)

    return np.column_stack((u1, u2, u3))


def sample_unit_sphere(frequency_x, frequency_y, frequency_z):
    """
    Generates evenly spaced sampled points on a unit sphere in R^3.

    Parameters:
    frequency_x (int): Sampling frequency along the x-axis.
    frequency_y (int): Sampling frequency along the y-axis.
    frequency_z (int): Sampling frequency along the z-axis.

    Returns:
    np.ndarray: An Nx3 numpy array containing the sampled points on the unit sphere,
                where N is the total number of sampled points.
    """
    u = np.linspace(0, 2 * np.pi, frequency_x)
    v = np.linspace(0, np.pi, frequency_y)
    u, v = np.meshgrid(u, v)

    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    indices = np.random.choice(points.shape[0], frequency_z, replace=False)
    sampled_points = points[indices]

    return sampled_points
