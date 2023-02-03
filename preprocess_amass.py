from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, glob
import numpy as np
from tqdm import tqdm
import torch
import smplx
from scipy.spatial.transform import Rotation as R
import json

def get_new_coordinate(body_mesh_model, betas, transl, pose):
    '''
    Get new coordinate, transform from world coordinate to AMASS coordinate.

    Args:
        - body_mesh_model: initialized body model object, e.g. 'smplx'
        - betas: shape coefficient, get from first n dims of a PCA
        - transl: translation
        - pose: pose[:,:3] represents for global orientation, pose[:,3:] represents for body rotation in aa format

    Return: 
        - global_ori_new: new coordinate system
        - transl_new: each subsequent frame will subtract the translation of the first frame
    '''
    bodyconfig = {}
    bodyconfig['transl'] = torch.FloatTensor(transl)
    bodyconfig['global_orient'] = torch.FloatTensor(pose[:,:3])
    bodyconfig['body_pose'] = torch.FloatTensor(pose[:,3:])
    bodyconfig['betas'] = torch.FloatTensor(betas).unsqueeze(0)
    #smplxout can be smplout, smplhout in utils
    smplxout = body_mesh_model(**bodyconfig)
    joints = smplxout.joints.squeeze().detach().cpu().numpy()
    x_axis = joints[2,:] - joints[1,:]
    x_axis[-1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.array([0,0,1])
    # Return the cross product of two (arrays of) vectors.
    # The cross product of a and b in :math:R^3 is a vector perpendicular to both a and b
    y_axis = np.cross(z_axis, x_axis) 
    y_axis = y_axis/np.linalg.norm(y_axis)
    global_ori_new = np.stack([x_axis, y_axis, z_axis], axis=1)
    transl_new = transl

    return global_ori_new, transl_new

def get_body_model(type, gender, batch_size, device='cpu'):
    '''
    get the body_model object.

    Args:
        - type: smpl, smplx smplh and others. Refer to smplx tutorial
        - gender: male, female, neutral
        - batch_size: batch size, an positive integar
        - device: which device should the model be on

    Return: 
        - body_model: body model object
    '''
    body_model_path = './support_data/body_models'
    body_model = smplx.create(body_model_path, model_type=type,
                                    gender=gender, ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=batch_size
                                    )
    if device == 'cuda':
        return body_model.cuda()
    else:
        return body_model

# main logic starts

##1. set input output dataset paths
amass_dataset_path = './support_data/amass_npz'
output_dataset_path = './support_data/amass_processed'
# you can set amass_subsets manually, e.g. as below
# amass_subsets =['ACCAD']
amass_subsets = [x for x in os.listdir(amass_dataset_path)
                    if os.path.isdir(amass_dataset_path+'/'+x)]
# whole amass subsets are ['ACCAD', 'BMLhandball', 'BMLmovi', 'BioMotionLab_NTroje', 'DanceDB', 'DFaust_67', 'EKUT', 'Eyes_Japan_Dataset', 'HUMAN4D', 'HumanEva', 'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap', 'TotalCapture', 'Transitions_mocap', 'CMU']

##2. set subsequence length
len_subseq = 480 # 4seconds under 120fps

##3. set markers
# read the corresponding smplx verts indices as markers.
with open('./support_data/body_models/marker_related/CMU.json') as f:
        marker_cmu_41 = list(json.load(f)['markersets'][0]['indices'].values())

with open('./support_data/body_models/marker_related/SSM2.json') as f:
        marker_ssm_67 = list(json.load(f)['markersets'][0]['indices'].values())

##4. main loop to each subset in AMASS
for subset in amass_subsets:
    # subset refers to 'ACCAD' and etc.
    # take each .npz file e.g.'support_data/amass_npz/ACCAD/Female1General_c3d/A9 - lie t2_poses.npz'
    seqs = glob.glob(os.path.join(amass_dataset_path, subset, '*/*.npz'))
    outfolder = os.path.join(output_dataset_path, subset)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    print('-- processing subset {:s}'.format(subset))

    index_subseq = 0 # index subsequences for subsets separately


    # main loop to process each sequence
    for seq in tqdm(seqs):
        # read data in the .npz file
        if os.path.basename(seq) == 'shape.npz':
            continue
        data = dict(np.load(seq))

        # define body model according to gender
        bodymodel_batch = get_body_model('smplx',str(data['gender'].astype(str)),len_subseq,device='cuda')
        bodymodel_one = get_body_model('smplx',str(data['gender'].astype(str)),1)

        # read data
        transl_all = data['trans']
        pose_all = data['poses']
        betas = data['betas']

        # skip too short sequences
        n_frames = transl_all.shape[0]
        if n_frames < len_subseq:
            continue

        t = 0
        while t < n_frames:
            # get subsequence and setup IO
            outfilename = os.path.join(outfolder, 'subseq_{:05d}.npz'.format(index_subseq))
            transl = transl_all[t:t+len_subseq, :]
            pose = pose_all[t:t+len_subseq, :]
            data_out = {}

            # drop last, break if remaining frames are not sufficient
            if transl.shape[0] < len_subseq:
                break

            # perform transformation from the world coordinate to the amass coordinate
            # get transformation for amass coordinate to world coordinate
            transf_rotmat, transf_transl = get_new_coordinate(bodymodel_one, betas[:10], transl[:1,:], pose[:1,:66])
            # get new global_orient
            global_ori = R.from_rotvec(pose[:,:3]).as_matrix() # to [t,3,3] rotation mat
            global_ori_new = np.einsum('ij,tjk->tik', transf_rotmat.T, global_ori)
            pose[:,:3] = R.from_matrix(global_ori_new).as_rotvec()
            # get new transl
            transl = np.einsum('ij,tj->ti', transf_rotmat.T, transl-transf_transl)
            # push to the result placeholder
            data_out['transf_rotmat'] = transf_rotmat
            data_out['transf_transl'] = transf_transl
            data_out['trans'] = transl
            data_out['poses'] = pose
            data_out['betas'] = betas
            data_out['gender'] = data['gender'].astype(str)
            data_out['mocap_framerate'] = data['mocap_framerate']

            # under this new amass coordinate, extract the joints/markers' locations
            # when get generated joints/markers, perform IK, get smplx params, and transform back to world coord
            body_param = {}
            body_param['transl'] = torch.FloatTensor(transl).cuda() # transformed
            body_param['global_orient'] = torch.FloatTensor(pose[:,:3]).cuda() # transformed
            body_param['betas'] = torch.FloatTensor(betas[:10]).unsqueeze(0).repeat(len_subseq,1).cuda()
            body_param['body_pose'] = torch.FloatTensor(pose[:, 3:66]).cuda()
            smplxout = bodymodel_batch(return_verts=True, **body_param)
            # extract joints and markers
            joints = smplxout.joints[:,:22,:].detach().squeeze().cpu().numpy()
            markers_41 = smplxout.vertices[:,marker_cmu_41,:].detach().squeeze().cpu().numpy()
            markers_67 = smplxout.vertices[:,marker_ssm_67,:].detach().squeeze().cpu().numpy()
            data_out['joints'] = joints
            data_out['marker_cmu_41'] = markers_41
            data_out['marker_ssm2_67'] = markers_67
            # save subsequence
            np.savez(outfilename, **data_out)
            t = t+len_subseq
            index_subseq = index_subseq + 1