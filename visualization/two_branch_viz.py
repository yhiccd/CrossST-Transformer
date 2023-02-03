import numpy as np
import torch
from fk import SMPLH_NR_JOINTS, SMPLH_BODY_MEMBERS, SMPLH_CONNECT, SMPLHForwardKinematics
fk_engine = SMPLHForwardKinematics()
data = np.load('support_data/amass_npz/ACCAD/Female1General_c3d/A1 - Stand_poses.npz')
aa = data['poses'][0][:66]
aa = torch.from_numpy(aa).unsqueeze(0)
pos = fk_engine.from_aa(aa)

# Two branches below

# skeleton_predicted branch(use Kinematics(parents), offsets and aa to recover 3D joints locations) 
from fk import SMPLH_NR_JOINTS, SMPLH_BODY_MEMBERS, SMPLH_CONNECT, SMPLHForwardKinematics
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Forward Kinematics
fk_engine = SMPLHForwardKinematics()
data = np.load('support_data/amass_npz/ACCAD/Female1General_c3d/A1 - Stand_poses.npz')
aa = data['poses'][0][:66]
aa = torch.from_numpy(aa).unsqueeze(0)
pos = fk_engine.from_aa(aa)
pos = pos.squeeze(0).cpu()

## use plot to show img
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pos[:, 0].cpu(), pos[:, 1].cpu(), pos[:, 2].cpu())
plt.show()

# skeleton_regressor branch(produce mesh v first, then regress 3D joints locations from mesh v)
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from os import path as osp

## get the config
support_dir = 'support_data/'
subject_gender = 'female'
num_betas = 16
num_dmpls = 8
dmpl_fname = osp.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))
bm_fname = osp.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## get the bm instance
bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)

## prepare angle-axis data, just take 1 frame as example, so we have to expand the 0 dim to be 1 as it refers to frame
poses = torch.Tensor(data['poses'][0, 3:66]).to(comp_device)
poses = poses.unsqueeze(0)

## pose the body_model
body_pose = bm(pose_body = poses)

## get the joints by using the pre_defined J_regressor
pos_gt = body_pose.Jtr[0][:22]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pos_gt[:, 0].cpu(), pos_gt[:, 1].cpu(), pos_gt[:, 2].cpu())
plt.show()
