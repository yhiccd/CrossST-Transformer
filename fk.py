import numpy as np
import quaternion
import cv2
import torch

# This does not take into account root position.
H36M_JOINTS_TO_IGNORE = [5, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31] # 13
# major_joints = 19
H36M_MAJOR_JOINTS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27] # 19
H36M_NR_JOINTS = 32
# parents = 32 = n_joints
H36M_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]
joints_left=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
joints_right=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]

reduce_list = [0, 1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
H36M_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
H36M_NR_JOINTS_REDUCE = 19
H36M_PARENTS_REDUCE = [-1, 0, 1, 2, 3, 0, 6, 7, 8, 0, 12, 13, 14, 13, 17, 18, 13, 25, 26]
# H36M_JOINTS = ['pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle', 'spine3',
#                'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
#                'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']
H36M_BODY_MEMBERS = {
        'left_arm': {'joints': [10, 16, 17, 18], 'side': 'left'},
        'right_arm': {'joints': [10, 13, 14, 15], 'side': 'right'},
        'head': {'joints': [10, 11, 12], 'side': 'middle'},
        'torso': {'joints': [0, 9, 10], 'side': 'middle'},
        'left_leg': {'joints': [0, 1, 2, 3, 4], 'side': 'left'},
        'right_leg': {'joints': [0, 5, 6, 7, 8], 'side': 'right'},
        }
H36M_MEMBERS = {
        'left_arm': {'joints': [13, 25, 26, 27], 'side': 'left'},
        'right_arm': {'joints': [13, 17, 18, 19], 'side': 'right'},
        'head': {'joints': [13, 14, 15], 'side': 'middle'},
        'torso': {'joints': [0, 12, 13], 'side': 'middle'},
        'left_leg': {'joints': [0, 1, 2, 3, 4], 'side': 'left'},
        'right_leg': {'joints': [0, 6, 7, 8, 9], 'side': 'right'},
        }
temp_H36M_CONNECT= []
for i in H36M_BODY_MEMBERS.keys():
    temp_H36M_CONNECT.append([H36M_BODY_MEMBERS[i]['joints'][j:j+2] for j in range(0, len(H36M_BODY_MEMBERS[i]['joints']), 1)][:-1])
H36M_CONNECT = []
for i in range(len(temp_H36M_CONNECT)):
    for j in range(len(temp_H36M_CONNECT[i])):
        H36M_CONNECT.append(temp_H36M_CONNECT[i][j])

SMPL_NR_JOINTS = 24
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPL_JOINTS = ['pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle', 'spine3',
               'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
               'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand']
SMPL_JOINT_MAPPING = {i: x for i, x in enumerate(SMPL_JOINTS)}
SMPL_BODY_MEMBERS = {
        'left_arm': {'joints': [9, 13, 16, 18, 20, 22], 'side': 'left'},
        'right_arm': {'joints': [9, 14, 17, 19, 21, 23], 'side': 'right'},
        'head': {'joints': [9, 12, 15], 'side': 'middle'},
        'torso': {'joints': [0, 3, 6, 9], 'side': 'middle'},
        'left_leg': {'joints': [0, 1, 4, 7, 10], 'side': 'left'},
        'right_leg': {'joints': [0, 2, 5, 8, 11], 'side': 'right'},
        }
temp_SMPL_CONNECT= []
for i in SMPL_BODY_MEMBERS.keys():
    temp_SMPL_CONNECT.append([SMPL_BODY_MEMBERS[i]['joints'][j:j+2] for j in range(0, len(SMPL_BODY_MEMBERS[i]['joints']), 1)][:-1])
SMPL_CONNECT = []
for i in range(len(temp_SMPL_CONNECT)):
    for j in range(len(temp_SMPL_CONNECT[i])):
        SMPL_CONNECT.append(temp_SMPL_CONNECT[i][j])

SMPLH_NR_JOINTS = 22
SMPLH_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SMPLH_JOINTS = ['pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle', 'spine3',
               'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
               'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']
SMPLH_JOINT_MAPPING = {i: x for i, x in enumerate(SMPLH_JOINTS)}
SMPLH_BODY_MEMBERS = {
        'left_arm': {'joints': [9, 13, 16, 18, 20], 'side': 'left'},
        'right_arm': {'joints': [9, 14, 17, 19, 21], 'side': 'right'},
        'head': {'joints': [9, 12, 15], 'side': 'middle'},
        'torso': {'joints': [0, 3, 6, 9], 'side': 'middle'},
        'left_leg': {'joints': [0, 1, 4, 7, 10], 'side': 'left'},
        'right_leg': {'joints': [0, 2, 5, 8, 11], 'side': 'right'},
        }
temp_SMPLH_CONNECT= []
for i in SMPLH_BODY_MEMBERS.keys():
    temp_SMPLH_CONNECT.append([SMPLH_BODY_MEMBERS[i]['joints'][j:j+2] for j in range(0, len(SMPLH_BODY_MEMBERS[i]['joints']), 1)][:-1])
SMPLH_CONNECT = []
for i in range(len(temp_SMPLH_CONNECT)):
    for j in range(len(temp_SMPLH_CONNECT[i])):
        SMPLH_CONNECT.append(temp_SMPLH_CONNECT[i][j])

class ForwardKinematics(object):
    '''
    Basic FK Engine for recovering skeleton position.
    '''
    def __init__(self, offsets, parents, left_mult=False, norm_idx=None, no_root=True):
        '''
        init FK Engine.

        Args:
            - offsets: vectors(refer to limbs) calc from plain model template
            - parents: kinematics tree
            - left_mult: whether multiply from left
            - norm_idx: define which vertor offset to be the norm(length=1)
            - no_root: do not use a translation

        Return: 
            - ForwardKinematics object
        '''
        self.offsets = offsets
        if norm_idx is not None:
            self.offsets = self.offsets / torch.linalg.norm(self.offsets[norm_idx])
        self.parents = parents
        self.n_joints = len(parents)
        self.left_mult = left_mult
        self.no_root = no_root
        assert self.offsets.shape[0] == self.n_joints

    def fk(self, joint_angles):
        '''
        Perform forward kinematics.
        This requires joint angles to be in rotation matrix format.

        Args:
            - joint_angles: tensor in the shape of (frame, n_joints*3*3)

        Return: 
            - positions: the 3D joints positions as a an tensor in the shape of (frame, n_joints, 3)
        '''
        assert joint_angles.shape[-1] == self.n_joints * 9
        angles = joint_angles.reshape(-1, self.n_joints, 3, 3).cuda()
        n_frames = angles.shape[0]
        positions = torch.zeros(n_frames, self.n_joints, 3).cuda()
        rotations = torch.zeros(n_frames, self.n_joints, 3, 3).cuda()  # intermediate storage of global rotation matrices
        if self.left_mult:
            offsets = self.offsets.unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, n_joints, 3)
        else:
            offsets = self.offsets.unsqueeze(0).unsqueeze(-1).cuda()  # (1, n_joints, 3, 1)

        if self.no_root:
            angles[:, 0] = torch.eye(3)

        for j in range(self.n_joints):
            if self.parents[j] == -1:
                # for root joint, we don't consider any root translation
                positions[:, j] = 0.0
                rotations[:, j] = angles[:, j]
            else:
                # this is a regular joint, perform forward kinematics
                if self.left_mult:
                    positions[:, j] = torch.squeeze(torch.matmul(offsets[:, :, j], rotations[:, self.parents[j]])) + \
                                      positions[:, self.parents[j]]
                    rotations[:, j] = torch.matmul(angles[:, j], rotations[:, self.parents[j]])
                else:
                    positions[:, j] = torch.squeeze(torch.matmul(rotations[:, self.parents[j]], offsets[:, j])) + \
                                      positions[:, self.parents[j]]
                    rotations[:, j] = torch.matmul(rotations[:, self.parents[j]], angles[:, j])

        return positions

    def from_aa(self, joint_angles):
        '''
        Get 3D joint positions from angle axis representations in shape of (frame, n_joints*3).

        Args:
            - joint_angles: tensor in the shape of (frame, n_joints*3*3)

        Return: 
            - positions: the 3D joints positions as a an tensor in the shape of (frame, n_joints, 3)
        '''
        angles = joint_angles.reshape(-1, self.n_joints, 3)
        angles_rot = np.zeros(angles.shape + (3,))
        for i in range(angles.shape[0]):
            for j in range(self.n_joints):
                angles_rot[i, j] = cv2.Rodrigues(angles[i, j])[0] # angles_rot[i, j] = cv2.Rodrigues(angles[i, j].numpy())[0]
        return self.fk(torch.from_numpy(np.reshape(angles_rot, [-1, self.n_joints * 9])).float())

    def from_rotmat(self, joint_angles):
        '''
        Get joint positions from rotation matrix representations in shape of (frame, n_joints*3*3).

        Args:
            - joint_angles: tensor in the shape of (frame, n_joints*3*3)

        Return: 
            - positions: the 3D joints positions as a an tensor in the shape of (frame, n_joints, 3)
        '''
        return self.fk(joint_angles)

    def from_quat(self, joint_angles):
        '''
        Get joint positions from quaternion representations in shape of (frame, n_joints*3).

        Args:
            - joint_angles: tensor in the shape of (frame, n_joints*3*3)

        Return: 
            - positions: the 3D joints positions as a an tensor in the shape of (frame, n_joints, 3)
        '''
        qs = quaternion.from_float_array(np.reshape(joint_angles, [-1, self.n_joints, 4]))
        aa = quaternion.as_rotation_matrix(qs)
        return self.fk(np.reshape(aa, [-1, self.n_joints * 3]))

class H36MForwardKinematics(ForwardKinematics):
    """
    Forward Kinematics for the skeleton defined by H3.6M dataset.
    """
    def __init__(self):
        offsets = torch.tensor([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                [-1.32948591e+02, 0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, -4.42894612e+02, 0.00000000e+00],
                                [0.00000000e+00, -4.54206447e+02, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00, 1.62767078e+02],
                                [0.00000000e+00, 0.00000000e+00, 7.49994370e+01],
                                [1.32948826e+02, 0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, -4.42894413e+02, 0.00000000e+00],
                                [0.00000000e+00, -4.54206590e+02, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00, 1.62767426e+02],
                                [0.00000000e+00, 0.00000000e+00, 7.49999480e+01],
                                [0.00000000e+00, 1.00000000e-01, 0.00000000e+00],
                                [0.00000000e+00, 2.33383263e+02, 0.00000000e+00],
                                [0.00000000e+00, 2.57077681e+02, 0.00000000e+00],
                                [0.00000000e+00, 1.21134938e+02, 0.00000000e+00],
                                [0.00000000e+00, 1.15002227e+02, 0.00000000e+00],
                                [0.00000000e+00, 2.57077681e+02, 0.00000000e+00],
                                [0.00000000e+00, 1.51034226e+02, 0.00000000e+00],
                                [0.00000000e+00, 2.78882773e+02, 0.00000000e+00],
                                [0.00000000e+00, 2.51733451e+02, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00, 9.99996270e+01],
                                [0.00000000e+00, 1.00000188e+02, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, 2.57077681e+02, 0.00000000e+00],
                                [0.00000000e+00, 1.51031437e+02, 0.00000000e+00],
                                [0.00000000e+00, 2.78892924e+02, 0.00000000e+00],
                                [0.00000000e+00, 2.51728680e+02, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00, 9.99998880e+01],
                                [0.00000000e+00, 1.37499922e+02, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

        # normalize so that right thigh has length 1
        super(H36MForwardKinematics, self).__init__(offsets, H36M_PARENTS, norm_idx=7,
                                                    left_mult=True)

class SMPLForwardKinematics(ForwardKinematics):
    """
    Forward Kinematics for the skeleton defined by SMPL.
    """
    def __init__(self):
        # these are the 3d locations stored under `J` in the SMPL model npz file, needs to calc the offsets manually.
        offsets = torch.tensor([[-8.76308970e-04, -2.11418723e-01, 2.78211200e-02],
                                [7.04848876e-02, -3.01002533e-01, 1.97749280e-02],
                                [-6.98883278e-02, -3.00379160e-01, 2.30254335e-02],
                                [-3.38451650e-03, -1.08161861e-01, 5.63597909e-03],
                                [1.01153808e-01, -6.65211904e-01, 1.30860155e-02],
                                [-1.06040718e-01, -6.71029623e-01, 1.38401121e-02],
                                [1.96440985e-04, 1.94957852e-02, 3.92296547e-03],
                                [8.95999143e-02, -1.04856032e+00, -3.04155922e-02],
                                [-9.20120818e-02, -1.05466743e+00, -2.80514913e-02],
                                [2.22362284e-03, 6.85680141e-02, 3.17901760e-02],
                                [1.12937580e-01, -1.10320516e+00, 8.39545265e-02],
                                [-1.14055299e-01, -1.10107698e+00, 8.98482216e-02],
                                [2.60992373e-04, 2.76811197e-01, -1.79753042e-02],
                                [7.75218998e-02, 1.86348444e-01, -5.08464100e-03],
                                [-7.48091986e-02, 1.84174211e-01, -1.00204779e-02],
                                [3.77815350e-03, 3.39133394e-01, 3.22299558e-02],
                                [1.62839013e-01, 2.18087461e-01, -1.23774789e-02],
                                [-1.64012068e-01, 2.16959041e-01, -1.98226746e-02],
                                [4.14086325e-01, 2.06120683e-01, -3.98959248e-02],
                                [-4.10001734e-01, 2.03806676e-01, -3.99843890e-02],
                                [6.52105424e-01, 2.15127546e-01, -3.98521818e-02],
                                [-6.55178550e-01, 2.12428626e-01, -4.35159074e-02],
                                [7.31773168e-01, 2.05445019e-01, -5.30577698e-02],
                                [-7.35578759e-01, 2.05180646e-01, -5.39352281e-02]])

        # need to convert them to compatible offsets
        smpl_offsets = torch.zeros([24, 3])
        smpl_offsets[0] = offsets[0]
        for idx, pid in enumerate(SMPL_PARENTS[1:]):
            smpl_offsets[idx+1] = offsets[idx + 1] - offsets[pid]

        # normalize so that right thigh has length 1
        super(SMPLForwardKinematics, self).__init__(smpl_offsets, SMPL_PARENTS, norm_idx=4,
                                                    left_mult=False)

class SMPLHForwardKinematics(ForwardKinematics):
    """
    Forward Kinematics for the skeleton defined by SMPL-H.
    """
    def __init__(self):
        # these are the 3d locations stored under `J` in the SMPLH model npz file, needs to calc the offsets manually.
        offsets = torch.tensor([[-8.76308970e-04, -2.11418723e-01, 2.78211200e-02],
                                [7.04848876e-02, -3.01002533e-01, 1.97749280e-02],
                                [-6.98883278e-02, -3.00379160e-01, 2.30254335e-02],
                                [-3.38451650e-03, -1.08161861e-01, 5.63597909e-03],
                                [1.01153808e-01, -6.65211904e-01, 1.30860155e-02],
                                [-1.06040718e-01, -6.71029623e-01, 1.38401121e-02],
                                [1.96440985e-04, 1.94957852e-02, 3.92296547e-03],
                                [8.95999143e-02, -1.04856032e+00, -3.04155922e-02],
                                [-9.20120818e-02, -1.05466743e+00, -2.80514913e-02],
                                [2.22362284e-03, 6.85680141e-02, 3.17901760e-02],
                                [1.12937580e-01, -1.10320516e+00, 8.39545265e-02],
                                [-1.14055299e-01, -1.10107698e+00, 8.98482216e-02],
                                [2.60992373e-04, 2.76811197e-01, -1.79753042e-02],
                                [7.75218998e-02, 1.86348444e-01, -5.08464100e-03],
                                [-7.48091986e-02, 1.84174211e-01, -1.00204779e-02],
                                [3.77815350e-03, 3.39133394e-01, 3.22299558e-02],
                                [1.62839013e-01, 2.18087461e-01, -1.23774789e-02],
                                [-1.64012068e-01, 2.16959041e-01, -1.98226746e-02],
                                [4.14086325e-01, 2.06120683e-01, -3.98959248e-02],
                                [-4.10001734e-01, 2.03806676e-01, -3.99843890e-02],
                                [6.52105424e-01, 2.15127546e-01, -3.98521818e-02],
                                [-6.55178550e-01, 2.12428626e-01, -4.35159074e-02]])

        # need to convert them to compatible offsets
        smplh_offsets = torch.zeros([22, 3])
        smplh_offsets[0] = offsets[0]
        for idx, pid in enumerate(SMPLH_PARENTS[1:]):
            smplh_offsets[idx+1] = offsets[idx + 1] - offsets[pid]

        # normalize so that right thigh has length 1
        super(SMPLHForwardKinematics, self).__init__(smplh_offsets, SMPLH_PARENTS, norm_idx=4,
                                                    left_mult=False)

class SMPLXForwardKinematics(ForwardKinematics):
    """
    Forward Kinematics for the skeleton defined by SMPL-X.
    """
    def __init__(self):
        # these are the 3d locations for SMPLH, will update later for SMPLX
        offsets = torch.tensor([[-8.76308970e-04, -2.11418723e-01, 2.78211200e-02],
                                [7.04848876e-02, -3.01002533e-01, 1.97749280e-02],
                                [-6.98883278e-02, -3.00379160e-01, 2.30254335e-02],
                                [-3.38451650e-03, -1.08161861e-01, 5.63597909e-03],
                                [1.01153808e-01, -6.65211904e-01, 1.30860155e-02],
                                [-1.06040718e-01, -6.71029623e-01, 1.38401121e-02],
                                [1.96440985e-04, 1.94957852e-02, 3.92296547e-03],
                                [8.95999143e-02, -1.04856032e+00, -3.04155922e-02],
                                [-9.20120818e-02, -1.05466743e+00, -2.80514913e-02],
                                [2.22362284e-03, 6.85680141e-02, 3.17901760e-02],
                                [1.12937580e-01, -1.10320516e+00, 8.39545265e-02],
                                [-1.14055299e-01, -1.10107698e+00, 8.98482216e-02],
                                [2.60992373e-04, 2.76811197e-01, -1.79753042e-02],
                                [7.75218998e-02, 1.86348444e-01, -5.08464100e-03],
                                [-7.48091986e-02, 1.84174211e-01, -1.00204779e-02],
                                [3.77815350e-03, 3.39133394e-01, 3.22299558e-02],
                                [1.62839013e-01, 2.18087461e-01, -1.23774789e-02],
                                [-1.64012068e-01, 2.16959041e-01, -1.98226746e-02],
                                [4.14086325e-01, 2.06120683e-01, -3.98959248e-02],
                                [-4.10001734e-01, 2.03806676e-01, -3.99843890e-02],
                                [6.52105424e-01, 2.15127546e-01, -3.98521818e-02],
                                [-6.55178550e-01, 2.12428626e-01, -4.35159074e-02]])

        # need to convert them to compatible offsets
        smplh_offsets = torch.zeros([22, 3])
        smplh_offsets[0] = offsets[0]
        for idx, pid in enumerate(SMPLH_PARENTS[1:]):
            smplh_offsets[idx+1] = offsets[idx + 1] - offsets[pid]

        # normalize so that right thigh has length 1
        super(SMPLHForwardKinematics, self).__init__(smplh_offsets, SMPLH_PARENTS, norm_idx=4,
                                                    left_mult=False)