import os
import math
import ipdb
import numpy as np
import cv2
import quaternion
import pickle
import torch
from tqdm import tqdm
from fk import H36MForwardKinematics

RNG = np.random.RandomState(42) # final answer of the universe :)

def read_txt_as_float(filename):
    """
    Reads a txt and returns a float matrix.

    Args:
        - filename: string. Path to the txt file
    
    Returns:
        - returnArray: the read data in a float32 matrix
    """
    # simply use the down below code, works as well
    data = np.loadtxt(filename, delimiter = ',', dtype = np.float32)
    return data


# set input output dataset paths
h36m_dataset_path = './H36M_related/h3.6m/dataset'
output_dataset_path = './H36M_related/h3.6m/dataset_3d'
all_subjects = [1, 6, 7, 8, 9, 11, 5]
actions = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]
reduce_list = [0, 1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
fk_engine = H36MForwardKinematics()
poses = []
# changed from file_ids to file_names
file_names = []
for subject in all_subjects:
    for action_idx in np.arange(len(actions)):
        
        action = actions[action_idx]

        for subact in [1, 2]: # subactions

            print("Processing subject {0}, action {1}, subaction {2}".format(subject, action, subact))

            filename = '{0}/S{1}/{2}_{3}.txt'.format(h36m_dataset_path, subject, action, subact)
            action_sequence = read_txt_as_float(filename)

            # remove the first three dimensions (root position) and the unwanted joints
            action_sequence = action_sequence[:, 3:] # (3476, 96)
            
            # transition
            positions = fk_engine.from_aa(action_sequence)

            positions = positions[:, reduce_list, :] # (3476, 19, 3)
            positions = np.reshape(positions.cpu().numpy(), [-1, len(reduce_list) * 3]) # (3476, 57)
            n_samples, dof = positions.shape # frames in total
            n_joints = dof // 3 # joints in total

            # store in record
            out_dir = '{0}/S{1}'.format(output_dataset_path, subject)
            out_filename = '{0}/S{1}/{2}_{3}.npz'.format(output_dataset_path, subject, action, subact)
            print("Store in to {0}, total frames {1}".format(out_dir, n_samples))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            np.savez(out_filename, positions = positions)