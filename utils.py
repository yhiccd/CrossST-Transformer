import os
import numpy as np
import quaternion
import cv2
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import numpy as np
from collections import namedtuple
from typing import Any
import math
import warnings
import glob
INTERVAL = [2, 4, 8, 10, 14, 25]
# @ 25 FPS, in ms: 80, 160, 320, 400, 560, 1000
METRIC_TARGET_LENGTHS_AMASS = [3, 6, 12, 18, 24, 36, 48, 60]
METRIC_PCK_THRESHS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

def files(curr_dir = '.', ext = '*.exe'):
    """all files"""
    for i in glob.glob(os.path.join(curr_dir, ext)):
        yield i

def all_files(rootdir, ext):
    """all files and sub dir"""
    for name in os.listdir(rootdir):
        if os.path.isdir(os.path.join(rootdir, name)):
            try:
                for i in all_files(os.path.join(rootdir, name), ext):
                    yield i
            except:
                pass
    for i in files(rootdir, ext):
        yield i

def remove_files(rootdir, ext, show = False):
    """remove rootdir 's ext files"""
    for i in files(rootdir, ext):
        if show:
            print(i)
        os.remove(i)

def remove_all_files(rootdir, ext, show = False):
    """remove rootdir and its sub dir 's ext files"""
    for i in all_files(rootdir, ext):
        if show:
            print(i)
        os.remove(i)

def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return torch.from_numpy(iden).float().cuda()

def is_valid_rotmat(rotmats, thresh=1e-5):
    '''
    Checks if the rotation matrices are valid, i.e. R*R' == I and det(R) == 1.

    Args:
        - rotmats: Rotation matrix, A tensor of shape (..., 3, 3).
        - thresh: Numerical threshold.
    
    Return: 
        - bool: True if all rotation matrices are valid, False if at least one is not valid.
    '''
    # check whether we have a valid rotation matrix
    # rotmat transpose
    rotmats_t = rotmats.permute(tuple(range(len(rotmats.shape[:-2]))) + (-1, -2))
    is_orthogonal_inter = torch.abs(torch.matmul(rotmats, rotmats_t) - eye(3, rotmats.shape[:-2])) < thresh
    is_orthogonal = (is_orthogonal_inter == True).sum() == is_orthogonal_inter.nelement()
    det_is_one_inter = torch.abs(torch.linalg.det(rotmats) - 1.0) < thresh
    det_is_one = (det_is_one_inter == True).sum() == det_is_one_inter.nelement()
    return is_orthogonal and det_is_one

def rotmat2euler(rotmats):
    '''
    Converts rotation matrices to euler angles.

    Args:
        - rotmats: Rotation matrix, A tensor of shape (..., 3, 3).
    
    Return: 
        - eul: An tensor of shape (..., 3) containing the Euler angles for each rotation matrix in `rotmats`
    '''
    assert rotmats.shape[-1] == 3 and rotmats.shape[-2] == 3
    orig_shape = rotmats.shape[:-2]
    rs = rotmats.reshape(-1, 3, 3)
    n_samples = rs.shape[0]
    
    # initialize to zeros
    e1 = torch.zeros([n_samples]).cuda()
    e2 = torch.zeros([n_samples]).cuda()
    e3 = torch.zeros([n_samples]).cuda()
    
    # find indices where we need to treat special cases
    is_one = rs[:, 0, 2] == 1
    is_minus_one = rs[:, 0, 2] == -1
    is_special = is_one | is_minus_one
    
    e1[is_special] = torch.atan2(rs[is_special, 0, 1], rs[is_special, 0, 2])
    e2[is_minus_one] = torch.pi/2
    e2[is_one] = -torch.pi/2
    
    # normal cases
    is_normal = ~(is_one | is_minus_one)
    # clip inputs to arcsin
    in_ = torch.clip(rs[is_normal, 0, 2], -1, 1)
    e2[is_normal] = -torch.asin(in_)
    e2_cos = torch.cos(e2[is_normal])
    e1[is_normal] = torch.atan2(rs[is_normal, 1, 2]/e2_cos,
                               rs[is_normal, 2, 2]/e2_cos)
    e3[is_normal] = torch.atan2(rs[is_normal, 0, 1]/e2_cos,
                               rs[is_normal, 0, 0]/e2_cos)
    
    eul = torch.stack([e1, e2, e3], dim=-1)
    eul = eul.reshape(orig_shape + eul.shape[1:])
    return eul

def quat2euler(quats, epsilon=0):
    '''
    Converts quats to euler angles.
    PSA: This function assumes Tait-Bryan angles, i.e. consecutive rotations rotate around the rotated coordinate
    system. Use at your own peril.

    Args:
        - quats: Quaternion numpy array of shape (..., 4).
    
    Return: 
        - eul: An np array of shape (..., 3) containing the Euler angles for each Quaternion in `quats`
    '''
    assert quats.shape[-1] == 4

    orig_shape = list(quats.shape)
    orig_shape[-1] = 3
    quats = np.reshape(quats, [-1, 4])

    q0 = quats[:, 0]
    q1 = quats[:, 1]
    q2 = quats[:, 2]
    q3 = quats[:, 3]

    x = np.arctan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    y = np.arcsin(np.clip(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
    z = np.arctan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))

    eul = np.stack([x, y, z], axis=-1)
    return np.reshape(eul, orig_shape)


def aa2rotmat(angle_axis):
    '''
    Convert angle-axis to rotation matrices using opencv's Rodrigues formula.
    
    Args:
        - angle_axis: A np array of shape (..., 3).
    
    Return: 
        - A np array of shape (..., 3, 3)
    '''
    orig_shape = angle_axis.shape[:-1]
    aas = np.reshape(angle_axis, [-1, 3])
    rots = np.zeros([aas.shape[0], 3, 3])
    for i in range(aas.shape[0]):
        rots[i] = cv2.Rodrigues(aas[i])[0]
    return np.reshape(rots, orig_shape + (3, 3))

def rotmat2aa_(rotmats):
    '''
    Convert rotation matrices to angle-axis using opencv's Rodrigues formula.
    
    Args:
        - rotmats: A np array of shape (..., 3, 3).
    
    Return: 
        - A np array of shape (..., 3)
    '''
    assert rotmats.shape[-1] == 3 and rotmats.shape[-2] == 3 and len(rotmats.shape) >= 3, 'invalid input dimension'
    orig_shape = rotmats.shape[:-2]
    rots = np.reshape(rotmats, [-1, 3, 3])
    aas = np.zeros([rots.shape[0], 3])
    for i in range(rots.shape[0]):
        aas[i] = np.squeeze(cv2.Rodrigues(rots[i])[0])
    return np.reshape(aas, orig_shape + (3,))

def correct_antipodal_quaternions(quat):
    '''
    Removes discontinuities coming from antipodal representation of quaternions
    At time step t it checks with representation, q or -q, is closer to time step t-1
    and choose the closest one.
    
    Args:
        - quat: numpy array of shape (N, K, 4) where N is the number of frames and K is the number of joints. K can be 0.
    
    Return: 
        - A numpy array of shape(N, K, 4) with fixed antipodal representation
    '''
    assert len(quat.shape) == 3 or len(quat.shape) == 2
    assert quat.shape[-1] == 4

    if len(quat.shape) == 2:
        quat_r = quat[:, np.newaxis].copy()
    else:
        quat_r = quat.copy()
    
    def dist(x, y):
        return np.sqrt(np.sum((x - y) ** 2, axis=-1))
    
    # Naive implementation looping over all time steps sequentially.
    # For a faster implementation check the QuaterNet paper.
    quat_corrected = np.zeros_like(quat_r)
    quat_corrected[0] = quat_r[0]
    for t in range(1, quat.shape[0]):
        diff_to_plus = dist(quat_r[t], quat_corrected[t - 1])
        diff_to_neg = dist(-quat_r[t], quat_corrected[t - 1])

        # diffs are vectors
        qc = quat_r[t]
        swap_idx = np.where(diff_to_neg < diff_to_plus)
        qc[swap_idx] = -quat_r[t, swap_idx]
        quat_corrected[t] = qc
    quat_corrected = np.squeeze(quat_corrected)
    return quat_corrected

def rotmat2quat(rotmats):
    '''
    Convert rotation matrices to quaternions.
    It ensures that there's no switch to the antipodal representation
    within this sequence of rotations.
    
    Args:
        - rotmats: A np array of shape (seq_length, n_joints*9).
    
    Return: 
        - A np array of shape (seq_length, n_joints*4).
    '''
    seq_length = rotmats.shape[0]
    assert rotmats.shape[1] % 9 == 0
    ori = np.reshape(rotmats, [seq_length, -1, 3, 3])
    ori_q = quaternion.as_float_array(quaternion.from_rotation_matrix(ori))
    ori_qc = correct_antipodal_quaternions(ori_q)
    ori_qc = np.reshape(ori_qc, [seq_length, -1])
    return ori_qc

def rotmat2aa(rotmats):
    '''
    Convert rotation matrices to angle-axis using opencv's Rodrigues formula.
    
    Args:
        - rotmats: A np array of shape (seq_length, n_joints*9)
    
    Return: 
        - A np array of shape (seq_length, n_joints*3)
    '''
    seq_length = rotmats.shape[0]
    assert rotmats.shape[1] % 9 == 0
    n_joints = rotmats.shape[1] // 9
    ori = np.reshape(rotmats, [seq_length*n_joints, 3, 3])
    aas = np.zeros([seq_length*n_joints, 3])
    for i in range(ori.shape[0]):
        aas[i] = np.squeeze(cv2.Rodrigues(ori[i])[0])
    return np.reshape(aas, [seq_length, n_joints*3])

def get_closest_rotmat(rotmats):
    '''
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    
    Args:
        - rotmats: A tensor of shape (..., 3, 3).
    
    Return: 
        - A tensor of the same shape as the inputs.
    '''
    u, s, vh = torch.linalg.svd(rotmats)
    r_closest = torch.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = torch.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = torch.sign(det)
    r_closest = torch.matmul(torch.matmul(u, iden), vh)
    return r_closest

def local_rot_to_global(joint_angles, parents, rep="rotmat", left_mult=False):
    '''
    Converts local rotations into global rotations by "unrolling" the kinematic chain.

    Args:
        - joint_angles: An tensor of rotation matrices of shape (N, nr_joints*dof)
        - parents: A tensor specifying the parent for each joint
        - rep: Which representation is used for `joint_angles`
        - left_mult: If True the local matrix is multiplied from the left, rather than the right
    
    Return: 
        - The global rotations as an tensor of rotation matrices in format (N, nr_joints, 3, 3)
    '''
    assert rep in ["rotmat", "quat", "aa"]
    n_joints = len(parents)
    if rep == "rotmat":
        rots = joint_angles.reshape(-1, n_joints, 3, 3)
    elif rep == "quat":
        rots = quaternion.as_rotation_matrix(quaternion.from_float_array(
            np.reshape(joint_angles, [-1, n_joints, 4])))
    else:
        rots = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(
            np.reshape(joint_angles, [-1, n_joints, 3])))

    out = torch.zeros_like(rots)
    num_joints = rots.shape[-3]
    for j in range(num_joints):
        if parents[j] < 0:
            # root rotation
            out[..., j, :, :] = rots[..., j, :, :]
        else:
            parent_rot = out[..., parents[j], :, :]
            local_rot = rots[..., j, :, :]
            lm = local_rot if left_mult else parent_rot
            rm = parent_rot if left_mult else local_rot
            out[..., j, :, :] = torch.matmul(lm, rm)
    return out

def save_model(model, model_dir, epoch=None):
    '''
    save model to a given dir

    Args:
        - model: the model
        - model_dir: save dir
        - epoch: epoch
    
    '''
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_Cross-Trans.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)

def load_model(model_dir, epoch=None):
    '''
    load model from a given dir

    Args:
        - model_dir: save dir
        - epoch: epoch
    
    Returns:
        - model
    '''
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_Cross-Trans.pt')
    if not os.path.exists(model_dir):
        return 
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class ModelWrapper(torch.nn.Module):
    '''
    Wrapper class for model with dict/list rvalues.
    '''

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model

    def forward(self, input_x: torch.Tensor, mask_x: torch.Tensor) -> Any:
        """
        Wrap forward call.
        """
        data = self.model(input_x, mask_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data