# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Camera transformation helper code.
"""

import math
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

_EPS = np.finfo(float).eps * 4.0


def unit_vector(data: NDArray, axis: Optional[int] = None) -> np.ndarray:
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    Args:
        axis: the axis along which to normalize into unit vector
        out: where to write out the data to. If None, returns a new np ndarray
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
        data /= math.sqrt(np.dot(data, data))
        return data
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    return data


def quaternion_from_matrix(matrix: NDArray, isprecise: bool = False) -> np.ndarray:
    """Return quaternion from rotation matrix.

    Args:
        matrix: rotation matrix to obtain quaternion
        isprecise: if True, input matrix is assumed to be precise rotation matrix and a faster algorithm is used.
    """
    # M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    M = np.asarray(matrix, dtype=np.float64)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
        K = np.array(K)
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[np.array([3, 0, 1, 2]), np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_slerp(
    quat0: NDArray, quat1: NDArray, fraction: float, spin: int = 0, shortestpath: bool = True
) -> np.ndarray:
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if q0 is None or q1 is None:
        raise ValueError("Input quaternions invalid.")
    if fraction == 0.0:
        return q0
    if fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def quaternion_matrix(quaternion: NDArray) -> np.ndarray:
    """Return homogeneous rotation matrix from quaternion.

    Args:
        quaternion: value to convert to matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def get_interpolated_poses(pose_a: NDArray, pose_b: NDArray, steps: int = 10) -> List[float]:
    """Return interpolation of poses with specified number of steps.
    Args:
        pose_a: first pose
        pose_b: second pose
        steps: number of steps the interpolated pose path should contain
    """

    quat_a = quaternion_from_matrix(pose_a[:3, :3])
    quat_b = quaternion_from_matrix(pose_b[:3, :3])

    ts = np.linspace(0, 1, steps)
    quats = [quaternion_slerp(quat_a, quat_b, t) for t in ts]
    trans = [(1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3] for t in ts]

    poses_ab = []
    for quat, tran in zip(quats, trans):
        pose = np.identity(4)
        pose[:3, :3] = quaternion_matrix(quat)[:3, :3]
        pose[:3, 3] = tran
        poses_ab.append(torch.from_numpy(pose[:3]))
    return poses_ab


def get_interpolated_k(
    k_a: Float[Tensor, "3 3"], k_b: Float[Tensor, "3 3"], steps: int = 10
) -> List[Float[Tensor, "3 4"]]:
    """
    Returns interpolated path between two camera poses with specified number of steps.

    Args:
        k_a: camera matrix 1
        k_b: camera matrix 2
        steps: number of steps the interpolated pose path should contain

    Returns:
        List of interpolated camera poses
    """
    Ks: List[Float[Tensor, "3 3"]] = []
    ts = np.linspace(0, 1, steps)
    for t in ts:
        new_k = k_a * (1.0 - t) + k_b * t
        Ks.append(new_k)
    return Ks


def get_ordered_poses_and_k(
    poses: Float[Tensor, "num_poses 3 4"],
    Ks: Float[Tensor, "num_poses 3 3"],
) -> Tuple[Float[Tensor, "num_poses 3 4"], Float[Tensor, "num_poses 3 3"]]:
    """
    Returns ordered poses and intrinsics by euclidian distance between poses.

    Args:
        poses: list of camera poses
        Ks: list of camera intrinsics

    Returns:
        tuple of ordered poses and intrinsics

    """

    poses_num = len(poses)

    ordered_poses = torch.unsqueeze(poses[0], 0)
    ordered_ks = torch.unsqueeze(Ks[0], 0)

    # remove the first pose from poses
    poses = poses[1:]
    Ks = Ks[1:]

    for _ in range(poses_num - 1):
        distances = torch.norm(ordered_poses[-1][:, 3] - poses[:, :, 3], dim=1)
        idx = torch.argmin(distances)
        ordered_poses = torch.cat((ordered_poses, torch.unsqueeze(poses[idx], 0)), dim=0)
        ordered_ks = torch.cat((ordered_ks, torch.unsqueeze(Ks[idx], 0)), dim=0)
        poses = torch.cat((poses[0:idx], poses[idx + 1 :]), dim=0)
        Ks = torch.cat((Ks[0:idx], Ks[idx + 1 :]), dim=0)

    return ordered_poses, ordered_ks


def get_interpolated_poses_many(
    poses: Float[Tensor, "num_poses 3 4"],
    Ks: Float[Tensor, "num_poses 3 3"],
    steps_per_transition: int = 10,
    order_poses: bool = False,
) -> Tuple[Float[Tensor, "num_poses 3 4"], Float[Tensor, "num_poses 3 3"]]:
    """Return interpolated poses for many camera poses.

    Args:
        poses: list of camera poses
        Ks: list of camera intrinsics
        steps_per_transition: number of steps per transition
        order_poses: whether to order poses by euclidian distance

    Returns:
        tuple of new poses and intrinsics
    """
    traj = []
    k_interp = []

    if order_poses:
        poses, Ks = get_ordered_poses_and_k(poses, Ks)

    for idx in range(poses.shape[0] - 1):
        pose_a = poses[idx].cpu().numpy()
        pose_b = poses[idx + 1].cpu().numpy()
        poses_ab = get_interpolated_poses(pose_a, pose_b, steps=steps_per_transition)
        traj += poses_ab
        k_interp += get_interpolated_k(Ks[idx], Ks[idx + 1], steps=steps_per_transition)

    traj = torch.stack(traj, dim=0).to(torch.float32)
    k_interp = torch.stack(k_interp, dim=0).to(torch.float32)

    return traj, k_interp


def normalize(x: torch.Tensor) -> Float[Tensor, "*batch"]:
    """Returns a normalized vector."""
    return x / torch.linalg.norm(x)


def viewmatrix(z, up, pos):
    """ Construct a camera view matrix using the look direction, up vector, and position. """
    vec2 = normalize(up)
    vec0 = normalize(torch.cross(z, vec2))
    vec1 = normalize(torch.cross(vec0, z))
    m = torch.stack([vec0, vec1, z, pos], 1)
    return m


# def get_forward_facing_trajectory(c2w, Ks, N):
#     '''
#     Args:
#         c2w: 4x4 tensor of camera-to-world matrices, in OpenCV format
#         Ks: 3x3 tensor of camera intrinsics,
#             [focal_x, 0, principal_x]
#             [0, focal_y, principal_y]
#             [0, 0, 0]
#             such as:
#             [523.3, 0, 452]
#             [0, 526.9, 256]
#             [0, 0, 0]
#         N: number of poses to generate
#     Usage:
#         generate a new trajectory of N poses facing forward
#         move forward along the z-axis in the camera coordinate system, and spiral around the central object

#     '''
#     poses = []
#     # Ks = Ks.cpu().numpy()  # Convert intrinsics to numpy for easier manipulation
#     Ks_list = []
#     for i in range(N):
#         # t = i / (N - 1)  # Interpolation factor
#         angle = 2.0 * np.pi * i / N  # Angle for the spiral
#         t = 0.7 + 0.7 * np.sin(angle)  # Interpolation factor with sinusoidal effect
#         # Move forward along the z-axis and introduce rotation
#         forward_translation = np.array([
#             [1, 0, 0, 0],  # Rotation around the z-axis
#             [0, 1, 0, 0],
#             [0, 0, 1, t],  # Forward movement
#             [0, 0, 0, 1]
#         ])
        
#         # Multiply the camera-to-world matrix with the new forward translation
#         new_pose = np.matmul(c2w.cpu().numpy(), forward_translation)
#         poses.append(torch.tensor(new_pose))
#         Ks_list.append(Ks)
#     return torch.stack(poses), torch.stack(Ks_list)
    
def normalize_np(x):
    return x / np.linalg.norm(x)

def get_forward_facing_trajectory(c2w, Ks, N, N_rots=2, zrate=0.25, focal=2.5):
    '''
    Args:
        c2w: 4x4 tensor of camera-to-world matrices, in OpenCV format
        Ks: 3x3 tensor of camera intrinsics,
            [focal_x, 0, principal_x]
            [0, focal_y, principal_y]
            [0, 0, 0]
            such as:
            [523.3, 0, 452]
            [0, 526.9, 256]
            [0, 0, 0]
        N: number of poses to generate
    Usage:
        generate a new trajectory of N poses facing forward
        move forward along the z-axis in the camera coordinate system, and spiral around the central object

    '''

    poses = []
    # Ks = Ks.cpu().numpy()  # Convert intrinsics to numpy for easier manipulation
    Ks_list = []
    rads = np.array([0.1, 0.1, 1.2, 1.])
    rads = np.array([0.12, 0.2, 0.7, 1.])
    c2w = c2w.cpu().numpy()
    for theta in np.linspace(0., 2. * np.pi * N_rots, N+1)[:-1]:
        # c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), np.sin(0.5*np.pi*theta/(2. * np.pi * N_rots)), 1.]) * rads)
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), (0.2 + theta/(2. * np.pi * N_rots)), 1.]) * rads)

        new_z = normalize_np(np.dot(c2w[:3,:4], np.array([0,0, focal, 1.])) - c)
        new_x = normalize_np(np.cross(np.array([0,1,0]), new_z))
        new_y = normalize_np(np.cross(new_z, new_x))
        new_c2w = np.eye(4)
        new_c2w[:3,:3] = np.stack([new_x, new_y, new_z], 1)
        new_c2w[:3,3] = c
        poses.append(torch.tensor(new_c2w))
        Ks_list.append(Ks)
    # for i in range(N):
    #     # t = i / (N - 1)  # Interpolation factor
    #     angle = 2.0 * np.pi * i / N  # Angle for the spiral
    #     t = 0.7 + 0.7 * np.sin(angle)  # Interpolation factor with sinusoidal effect
    #     # Move forward along the z-axis and introduce rotation
    #     forward_translation = np.array([
    #         [1, 0, 0, 0],  # Rotation around the z-axis
    #         [0, 1, 0, 0],
    #         [0, 0, 1, t],  # Forward movement
    #         [0, 0, 0, 1]
    #     ])
        
    #     # Multiply the camera-to-world matrix with the new forward translation
    #     new_pose = np.matmul(c2w.cpu().numpy(), forward_translation)
    #     poses.append(torch.tensor(new_pose))
    #     Ks_list.append(Ks)
    return torch.stack(poses), torch.stack(Ks_list)
    


def normalize_with_norm(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize tensor along axis and return normalized value with norms.

    Args:
        x: tensor to normalize.
        dim: axis along which to normalize.

    Returns:
        Tuple of normalized tensor and corresponding norm.
    """

    norm = torch.maximum(torch.linalg.vector_norm(x, dim=dim, keepdims=True), torch.tensor([_EPS]).to(x))
    return x / norm, norm


def viewmatrix(lookat: torch.Tensor, up: torch.Tensor, pos: torch.Tensor) -> Float[Tensor, "*batch"]:
    """Returns a camera transformation matrix.

    Args:
        lookat: The direction the camera is looking.
        up: The upward direction of the camera.
        pos: The position of the camera.

    Returns:
        A camera transformation matrix.
    """
    vec2 = normalize(lookat)
    vec1_avg = normalize(up)
    vec0 = normalize(torch.cross(vec1_avg, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, pos], 1)
    return m


def get_distortion_params(
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Float[Tensor, "*batch"]:
    """Returns a distortion parameters matrix.

    Args:
        k1: The first radial distortion parameter.
        k2: The second radial distortion parameter.
        k3: The third radial distortion parameter.
        k4: The fourth radial distortion parameter.
        p1: The first tangential distortion parameter.
        p2: The second tangential distortion parameter.
    Returns:
        torch.Tensor: A distortion parameters matrix.
    """
    return torch.Tensor([k1, k2, k3, k4, p1, p2])


def _compute_residual_and_jacobian(
    x: torch.Tensor,
    y: torch.Tensor,
    xd: torch.Tensor,
    yd: torch.Tensor,
    distortion_params: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Auxiliary function of radial_and_tangential_undistort() that computes residuals and jacobians.
    Adapted from MultiNeRF:
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474

    Args:
        x: The updated x coordinates.
        y: The updated y coordinates.
        xd: The distorted x coordinates.
        yd: The distorted y coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].

    Returns:
        The residuals (fx, fy) and jacobians (fx_x, fx_y, fy_x, fy_y).
    """

    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    k3 = distortion_params[..., 2]
    k4 = distortion_params[..., 3]
    p1 = distortion_params[..., 4]
    p2 = distortion_params[..., 5]

    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


# @torch_compile(dynamic=True, mode="reduce-overhead", backend="eager")
def radial_and_tangential_undistort(
    coords: torch.Tensor,
    distortion_params: torch.Tensor,
    eps: float = 1e-3,
    max_iterations: int = 10,
) -> torch.Tensor:
    """Computes undistorted coords given opencv distortion parameters.
    Adapted from MultiNeRF
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509

    Args:
        coords: The distorted coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].
        eps: The epsilon for the convergence.
        max_iterations: The maximum number of iterations to perform.

    Returns:
        The undistorted coordinates.
    """

    # Initialize from the distorted point.
    x = coords[..., 0]
    y = coords[..., 1]

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=coords[..., 0], yd=coords[..., 1], distortion_params=distortion_params
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(torch.abs(denominator) > eps, x_numerator / denominator, torch.zeros_like(denominator))
        step_y = torch.where(torch.abs(denominator) > eps, y_numerator / denominator, torch.zeros_like(denominator))

        x = x + step_x
        y = y + step_y

    return torch.stack([x, y], dim=-1)


def rotation_matrix(a: Float[Tensor, "3"], b: Float[Tensor, "3"]) -> Float[Tensor, "3 3"]:
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


def focus_of_attention(poses: Float[Tensor, "*num_poses 4 4"], initial_focus: Float[Tensor, "3"]) -> Float[Tensor, "3"]:
    """Compute the focus of attention of a set of cameras. Only cameras
    that have the focus of attention in front of them are considered.

     Args:
        poses: The poses to orient.
        initial_focus: The 3D point views to decide which cameras are initially activated.

    Returns:
        The 3D position of the focus of attention.
    """
    # References to the same method in third-party code:
    # https://github.com/google-research/multinerf/blob/1c8b1c552133cdb2de1c1f3c871b2813f6662265/internal/camera_utils.py#L145
    # https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/load_llff.py#L197
    active_directions = -poses[:, :3, 2:3]
    active_origins = poses[:, :3, 3:4]
    # initial value for testing if the focus_pt is in front or behind
    focus_pt = initial_focus
    # Prune cameras which have the current have the focus_pt behind them.
    active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
    done = False
    # We need at least two active cameras, else fallback on the previous solution.
    # This may be the "poses" solution if no cameras are active on first iteration, e.g.
    # they are in an outward-looking configuration.
    while torch.sum(active.int()) > 1 and not done:
        active_directions = active_directions[active]
        active_origins = active_origins[active]
        # https://en.wikipedia.org/wiki/Line–line_intersection#In_more_than_two_dimensions
        m = torch.eye(3) - active_directions * torch.transpose(active_directions, -2, -1)
        mt_m = torch.transpose(m, -2, -1) @ m
        focus_pt = torch.linalg.inv(mt_m.mean(0)) @ (mt_m @ active_origins).mean(0)[:, 0]
        active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
        if active.all():
            # the set of active cameras did not change, so we're done.
            done = True
    return focus_pt


def auto_orient_and_center_poses(
    poses: Float[Tensor, "*num_poses 4 4"],
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
) -> Tuple[Float[Tensor, "*num_poses 3 4"], Float[Tensor, "3 4"]]:
    """Orients and centers the poses.

    We provide three methods for orientation:

    - pca: Orient the poses so that the principal directions of the camera centers are aligned
        with the axes, Z corresponding to the smallest principal component.
        This method works well when all of the cameras are in the same plane, for example when
        images are taken using a mobile robot.
    - up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.
    - vertical: Orient the poses so that the Z 3D direction projects close to the
        y axis in images. This method works better if cameras are not all
        looking in the same 3D direction, which may happen in camera arrays or in LLFF.

    There are two centering methods:

    - poses: The poses are centered around the origin.
    - focus: The origin is set to the focus of attention of all cameras (the
        closest point to cameras optical axes). Recommended for inward-looking
        camera configurations.

    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_method: The method to use to center the poses.

    Returns:
        Tuple of the oriented poses and the transform matrix.
    """

    origins = poses[..., :3, 3]

    mean_origin = torch.mean(origins, dim=0)
    translation_diff = origins - mean_origin

    if center_method == "poses":
        translation = mean_origin
    elif center_method == "focus":
        translation = focus_of_attention(poses, mean_origin)
    elif center_method == "none":
        translation = torch.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown value for center_method: {center_method}")

    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))

        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

        if oriented_poses.mean(dim=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]
    elif method in ("up", "vertical"):
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        if method == "vertical":
            # If cameras are not all parallel (e.g. not in an LLFF configuration),
            # we can find the 3D direction that most projects vertically in all
            # cameras by minimizing ||Xu|| s.t. ||u||=1. This total least squares
            # problem is solved by SVD.
            x_axis_matrix = poses[:, :3, 0]
            _, S, Vh = torch.linalg.svd(x_axis_matrix, full_matrices=False)
            # Singular values are S_i=||Xv_i|| for each right singular vector v_i.
            # ||S|| = sqrt(n) because lines of X are all unit vectors and the v_i
            # are an orthonormal basis.
            # ||Xv_i|| = sqrt(sum(dot(x_axis_j,v_i)^2)), thus S_i/sqrt(n) is the
            # RMS of cosines between x axes and v_i. If the second smallest singular
            # value corresponds to an angle error less than 10° (cos(80°)=0.17),
            # this is probably a degenerate camera configuration (typical values
            # are around 5° average error for the true vertical). In this case,
            # rather than taking the vector corresponding to the smallest singular
            # value, we project the "up" vector on the plane spanned by the two
            # best singular vectors. We could also just fallback to the "up"
            # solution.
            if S[1] > 0.17 * math.sqrt(poses.shape[0]):
                # regular non-degenerate configuration
                up_vertical = Vh[2, :]
                # It may be pointing up or down. Use "up" to disambiguate the sign.
                up = up_vertical if torch.dot(up_vertical, up) > 0 else -up_vertical
            else:
                # Degenerate configuration: project "up" on the plane spanned by
                # the last two right singular vectors (which are orthogonal to the
                # first). v_0 is a unit vector, no need to divide by its norm when
                # projecting.
                up = up - Vh[0, :] * torch.dot(up, Vh[0, :])
                # re-normalize
                up = up / torch.linalg.norm(up)

        rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
    elif method == "none":
        transform = torch.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError(f"Unknown value for method: {method}")

    return oriented_poses, transform


@torch.jit.script
def fisheye624_project(xyz, params):
    """
    Batched implementation of the FisheyeRadTanThinPrism (aka Fisheye624) camera
    model project() function.
    Inputs:
        xyz: BxNx3 tensor of 3D points to be projected
        params: Bx16 tensor of Fisheye624 parameters formatted like this:
                [f_u f_v c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
                or Bx15 tensor of Fisheye624 parameters formatted like this:
                [f c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
    Outputs:
        uv: BxNx2 tensor of 2D projections of xyz in image plane
    Model for fisheye cameras with radial, tangential, and thin-prism distortion.
    This model allows fu != fv.
    Specifically, the model is:
    uvDistorted = [x_r]  + tangentialDistortion  + thinPrismDistortion
                  [y_r]
    proj = diag(fu,fv) * uvDistorted + [cu;cv];
    where:
      a = x/z, b = y/z, r = (a^2+b^2)^(1/2)
      th = atan(r)
      cosPhi = a/r, sinPhi = b/r
      [x_r]  = (th+ k0 * th^3 + k1* th^5 + ...) [cosPhi]
      [y_r]                                     [sinPhi]
      the number of terms in the series is determined by the template parameter numK.
      tangentialDistortion = [(2 x_r^2 + rd^2)*p_0 + 2*x_r*y_r*p_1]
                             [(2 y_r^2 + rd^2)*p_1 + 2*x_r*y_r*p_0]
      where rd^2 = x_r^2 + y_r^2
      thinPrismDistortion = [s0 * rd^2 + s1 rd^4]
                            [s2 * rd^2 + s3 rd^4]
    Author: Daniel DeTone (ddetone@meta.com)
    """

    assert xyz.ndim == 3
    assert params.ndim == 2
    assert params.shape[-1] == 16 or params.shape[-1] == 15, "This model allows fx != fy"
    eps = 1e-9
    B, N = xyz.shape[0], xyz.shape[1]

    # Radial correction.
    z = xyz[:, :, 2].reshape(B, N, 1)
    z = torch.where(torch.abs(z) < eps, eps * torch.sign(z), z)
    ab = xyz[:, :, :2] / z
    r = torch.norm(ab, dim=-1, p=2, keepdim=True)
    th = torch.atan(r)
    th_divr = torch.where(r < eps, torch.ones_like(ab), ab / r)
    th_k = th.reshape(B, N, 1).clone()
    for i in range(6):
        th_k = th_k + params[:, -12 + i].reshape(B, 1, 1) * torch.pow(th, 3 + i * 2)
    xr_yr = th_k * th_divr
    uv_dist = xr_yr

    # Tangential correction.
    p0 = params[:, -6].reshape(B, 1)
    p1 = params[:, -5].reshape(B, 1)
    xr = xr_yr[:, :, 0].reshape(B, N)
    yr = xr_yr[:, :, 1].reshape(B, N)
    xr_yr_sq = torch.square(xr_yr)
    xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
    yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
    rd_sq = xr_sq + yr_sq
    uv_dist_tu = uv_dist[:, :, 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
    uv_dist_tv = uv_dist[:, :, 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)
    uv_dist = torch.stack([uv_dist_tu, uv_dist_tv], dim=-1)  # Avoids in-place complaint.

    # Thin Prism correction.
    s0 = params[:, -4].reshape(B, 1)
    s1 = params[:, -3].reshape(B, 1)
    s2 = params[:, -2].reshape(B, 1)
    s3 = params[:, -1].reshape(B, 1)
    rd_4 = torch.square(rd_sq)
    uv_dist[:, :, 0] = uv_dist[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
    uv_dist[:, :, 1] = uv_dist[:, :, 1] + (s2 * rd_sq + s3 * rd_4)

    # Finally, apply standard terms: focal length and camera centers.
    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)
    result = uv_dist * fx_fy + cx_cy

    return result


# Core implementation of fisheye 624 unprojection. More details are documented here:
# https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/camera_intrinsic_models#the-fisheye62-model
@torch.jit.script
def fisheye624_unproject_helper(uv, params, max_iters: int = 5):
    """
    Batched implementation of the FisheyeRadTanThinPrism (aka Fisheye624) camera
    model. There is no analytical solution for the inverse of the project()
    function so this solves an optimization problem using Newton's method to get
    the inverse.
    Inputs:
        uv: BxNx2 tensor of 2D pixels to be unprojected
        params: Bx16 tensor of Fisheye624 parameters formatted like this:
                [f_u f_v c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
                or Bx15 tensor of Fisheye624 parameters formatted like this:
                [f c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
    Outputs:
        xyz: BxNx3 tensor of 3D rays of uv points with z = 1.
    Model for fisheye cameras with radial, tangential, and thin-prism distortion.
    This model assumes fu=fv. This unproject function holds that:
    X = unproject(project(X))     [for X=(x,y,z) in R^3, z>0]
    and
    x = project(unproject(s*x))   [for s!=0 and x=(u,v) in R^2]
    Author: Daniel DeTone (ddetone@meta.com)
    """

    assert uv.ndim == 3, "Expected batched input shaped BxNx3"
    assert params.ndim == 2
    assert params.shape[-1] == 16 or params.shape[-1] == 15, "This model allows fx != fy"
    eps = 1e-6
    B, N = uv.shape[0], uv.shape[1]

    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)

    uv_dist = (uv - cx_cy) / fx_fy

    # Compute xr_yr using Newton's method.
    xr_yr = uv_dist.clone()  # Initial guess.
    for _ in range(max_iters):
        uv_dist_est = xr_yr.clone()
        # Tangential terms.
        p0 = params[:, -6].reshape(B, 1)
        p1 = params[:, -5].reshape(B, 1)
        xr = xr_yr[:, :, 0].reshape(B, N)
        yr = xr_yr[:, :, 1].reshape(B, N)
        xr_yr_sq = torch.square(xr_yr)
        xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
        yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
        rd_sq = xr_sq + yr_sq
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)
        # Thin Prism terms.
        s0 = params[:, -4].reshape(B, 1)
        s1 = params[:, -3].reshape(B, 1)
        s2 = params[:, -2].reshape(B, 1)
        s3 = params[:, -1].reshape(B, 1)
        rd_4 = torch.square(rd_sq)
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (s2 * rd_sq + s3 * rd_4)
        # Compute the derivative of uv_dist w.r.t. xr_yr.
        duv_dist_dxr_yr = uv.new_ones(B, N, 2, 2)
        duv_dist_dxr_yr[:, :, 0, 0] = 1.0 + 6.0 * xr_yr[:, :, 0] * p0 + 2.0 * xr_yr[:, :, 1] * p1
        offdiag = 2.0 * (xr_yr[:, :, 0] * p1 + xr_yr[:, :, 1] * p0)
        duv_dist_dxr_yr[:, :, 0, 1] = offdiag
        duv_dist_dxr_yr[:, :, 1, 0] = offdiag
        duv_dist_dxr_yr[:, :, 1, 1] = 1.0 + 6.0 * xr_yr[:, :, 1] * p1 + 2.0 * xr_yr[:, :, 0] * p0
        xr_yr_sq_norm = xr_yr_sq[:, :, 0] + xr_yr_sq[:, :, 1]
        temp1 = 2.0 * (s0 + 2.0 * s1 * xr_yr_sq_norm)
        duv_dist_dxr_yr[:, :, 0, 0] = duv_dist_dxr_yr[:, :, 0, 0] + (xr_yr[:, :, 0] * temp1)
        duv_dist_dxr_yr[:, :, 0, 1] = duv_dist_dxr_yr[:, :, 0, 1] + (xr_yr[:, :, 1] * temp1)
        temp2 = 2.0 * (s2 + 2.0 * s3 * xr_yr_sq_norm)
        duv_dist_dxr_yr[:, :, 1, 0] = duv_dist_dxr_yr[:, :, 1, 0] + (xr_yr[:, :, 0] * temp2)
        duv_dist_dxr_yr[:, :, 1, 1] = duv_dist_dxr_yr[:, :, 1, 1] + (xr_yr[:, :, 1] * temp2)
        # Compute 2x2 inverse manually here since torch.inverse() is very slow.
        # Because this is slow: inv = duv_dist_dxr_yr.inverse()
        # About a 10x reduction in speed with above line.
        mat = duv_dist_dxr_yr.reshape(-1, 2, 2)
        a = mat[:, 0, 0].reshape(-1, 1, 1)
        b = mat[:, 0, 1].reshape(-1, 1, 1)
        c = mat[:, 1, 0].reshape(-1, 1, 1)
        d = mat[:, 1, 1].reshape(-1, 1, 1)
        det = 1.0 / ((a * d) - (b * c))
        top = torch.cat([d, -b], dim=2)
        bot = torch.cat([-c, a], dim=2)
        inv = det * torch.cat([top, bot], dim=1)
        inv = inv.reshape(B, N, 2, 2)
        # Manually compute 2x2 @ 2x1 matrix multiply.
        # Because this is slow: step = (inv @ (uv_dist - uv_dist_est)[..., None])[..., 0]
        diff = uv_dist - uv_dist_est
        a = inv[:, :, 0, 0]
        b = inv[:, :, 0, 1]
        c = inv[:, :, 1, 0]
        d = inv[:, :, 1, 1]
        e = diff[:, :, 0]
        f = diff[:, :, 1]
        step = torch.stack([a * e + b * f, c * e + d * f], dim=-1)
        # Newton step.
        xr_yr = xr_yr + step

    # Compute theta using Newton's method.
    xr_yr_norm = xr_yr.norm(p=2, dim=2).reshape(B, N, 1)
    th = xr_yr_norm.clone()
    for _ in range(max_iters):
        th_radial = uv.new_ones(B, N, 1)
        dthd_th = uv.new_ones(B, N, 1)
        for k in range(6):
            r_k = params[:, -12 + k].reshape(B, 1, 1)
            th_radial = th_radial + (r_k * torch.pow(th, 2 + k * 2))
            dthd_th = dthd_th + ((3.0 + 2.0 * k) * r_k * torch.pow(th, 2 + k * 2))
        th_radial = th_radial * th
        step = (xr_yr_norm - th_radial) / dthd_th
        # handle dthd_th close to 0.
        step = torch.where(dthd_th.abs() > eps, step, torch.sign(step) * eps * 10.0)
        th = th + step
    # Compute the ray direction using theta and xr_yr.
    close_to_zero = torch.logical_and(th.abs() < eps, xr_yr_norm.abs() < eps)
    ray_dir = torch.where(close_to_zero, xr_yr, torch.tan(th) / xr_yr_norm * xr_yr)
    ray = torch.cat([ray_dir, uv.new_ones(B, N, 1)], dim=2)
    return ray


# unproject 2D point to 3D with fisheye624 model
def fisheye624_unproject(coords: torch.Tensor, distortion_params: torch.Tensor) -> torch.Tensor:
    dirs = fisheye624_unproject_helper(coords.unsqueeze(0), distortion_params[0].unsqueeze(0))
    # correct for camera space differences:
    dirs[..., 1] = -dirs[..., 1]
    dirs[..., 2] = -dirs[..., 2]
    return dirs


# custom functions for rustLVSM
from torch.nn import functional as F

def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # quaternions = F.normalize(quaternions, dim=-1)
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def pose_encoding_to_c2w(
    pose_encoding, # [B, V, 7] or [B, V, 8]
):
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]

    R = quat_to_mat(quat)
    c2w = torch.cat([R, T[..., None]], dim=-1) # [B, V, 3, 4]
    c2w = torch.cat([c2w, torch.zeros_like(c2w[..., :1, :])], dim=-2) # [B, V, 4, 4]
    c2w[..., 3, 3] = 1.0
    return c2w

def c2w_to_pose_encoding(c2w, pose_enc_type="absT_quaR"):
    if pose_enc_type == "absT_quaR_FoV":
        raise NotImplementedError
    
    R = c2w[..., :3, :3]
    T = c2w[..., :3, 3]
    quat = mat_to_quat(R)
    return torch.cat([T, quat], dim=-1)

