import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import cv2
import torch
from H36M_related.train_h36m import inference
from utils import load_model
from fk import SMPL_NR_JOINTS, SMPL_BODY_MEMBERS, SMPL_CONNECT, SMPLForwardKinematics
from fk import SMPLH_NR_JOINTS, SMPLH_BODY_MEMBERS, SMPLH_CONNECT, SMPLHForwardKinematics

data_dir = "/H36M_related/h3.6m/dataset"
train_subject = [1, 6, 7, 8, 9, 11]
test_subject = [5]
actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]

def read_txt_as_float(filename):
    """
    Reads a txt and returns a float matrix.

    Args:
        - filename: string. Path to the txt file
    
    Returns:
        - returnArray: the read data in a float32 matrix
    """
    # simply use the code down below, works as well
    # data = np.loadtxt(filename, delimiter = ',', dtype = np.float32)
    # return data
    out_array = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            out_array.append(np.array([np.float32(x) for x in line]))
    
    return np.array(out_array)

def draw_sequence(positions, body_member, global_connect, saveflag, tag, color_sk):
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.ion()
    for frame in range(positions.shape[0]):
        ax.clear()
        ax.grid(False)
        ax.axis('off')
        ax.view_init(90, -90)
        ax.set_xlim3d(-2, 1.5)
        ax.set_ylim3d(-2, 1.5)
        ax.set_zlim3d(-2, 1.5)
        
        for kinematics in body_member.keys():
            reduce_x = positions[frame][body_member[kinematics]['joints'], 0]
            reduce_y = positions[frame][body_member[kinematics]['joints'], 1]
            reduce_z = positions[frame][body_member[kinematics]['joints'], 2]
            ax.scatter(reduce_x, reduce_y, reduce_z, c = "r", label = kinematics, s = 10)


        for i in range(len(global_connect)):
            for j in range(len(global_connect[i])):
                ax.plot([positions[frame][global_connect[i][j][0], 0], positions[frame][global_connect[i][j][1], 0]],
                        [positions[frame][global_connect[i][j][0], 1], positions[frame][global_connect[i][j][1], 1]],
                        [positions[frame][global_connect[i][j][0], 2], positions[frame][global_connect[i][j][1], 2]], color = color_sk, linewidth = 5)
                # print("{} connect to {}".format(global_connect_reduce[i][j][0], global_connect_reduce[i][j][1]))
        
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})

        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        if saveflag:
            plt.savefig("pics/{}_{}.png".format(tag, frame))
        plt.pause(0.0001)
    plt.ioff()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print(torch.cuda.is_available())

    # prediction
    folder = 'your model'
    epo = 461
    action_name = 'walkingtogether'
    frame = 40
    obs = 50
    pred = 25
    model = load_model(folder, epo)

    print("Reading subject {0}, action {1}, subaction {2}".format(5, action_name, 1))
    poses = []
    filename = '{0}/S{1}/{2}_{3}.txt'.format(data_dir, 5, action_name, 1)
    action_sequence = read_txt_as_float(filename)

    # remove the first three dimensions (root position) and the unwanted joints
    path = action_sequence[:, :3]
    action_sequence = action_sequence[:, 3:] # (3476, 96)
    action_sequence = np.reshape(action_sequence, [-1, SMPLH_NR_JOINTS, 3]) # (3476, 32, 3)
    # action_sequence = action_sequence[:, H36M_USED_JOINTS] # (3476, 17, 3)
    action_sequence = np.reshape(action_sequence, [-1, SMPLH_NR_JOINTS * 3]) # H36M_NR_JOINTS -> 17 (3476, 51)
    n_samples, dof = action_sequence.shape # frames in total
    n_joints = dof // 3 # joints in total


    expmap = np.reshape(action_sequence, [n_samples * n_joints, 3]) # (number of all single joint)
    # first three values are positions, so technically it's meaningless to convert them,
    # but we'll do it anyway because later we discard this values anyhow
    rotmats = np.zeros([n_samples * n_joints, 3, 3])
    for i in range(rotmats.shape[0]):
        rotmats[i] = cv2.Rodrigues(expmap[i])[0]
    rotmats = np.reshape(rotmats, [n_samples, n_joints*3*3])
    action_sequence = rotmats


    # downsample to 25 fps
    even_list = range(0, n_samples, 2)
    poses.append(action_sequence[even_list, :])
    # file_names.append("S{}_{}_{}".format(subject, action, subact))
    poses = np.array(poses).squeeze()
    poses = np.reshape(poses, [-1, n_joints*3*3])
    # prediction
    obs_seq = poses[frame:frame+obs]# 275:325
    obs_seq = np.reshape(obs_seq, [1, 50, 32, 9])
    obs_seq_tensor = torch.from_numpy(obs_seq).type(torch.float32).cuda()
    inference_seq_tensor = inference(model, obs_seq_tensor, 50, 1, pred)
    inference_seq = inference_seq_tensor.squeeze().cpu().numpy()
    inference_seq = np.reshape(inference_seq, [-1, n_joints*3*3])
    # prediction
    # downsample to 25 fps
    path_25 = path[0:path.size:2]
    fk_engine = SMPLHForwardKinematics()
    positions = fk_engine.fk(poses)
    positions.shape
    # prediction
    predicted_positions = fk_engine.fk(inference_seq)
    draw_path = False

    if draw_path:
        fig_path = plt.figure()
        ax_path = Axes3D(fig_path)
        plt.ion()
        for frame in range(path.shape[0]):
            path_x = path[frame][0]
            path_y = path[frame][1]
            path_z = path[frame][2]
            ax_path.scatter(path_x, path_y, path_z, c = "b", label = str(frame))
            plt.pause(3)
        plt.ioff()

        # 275:325, 325:350
        draw_sequence(positions[frame:frame+obs], SMPLH_BODY_MEMBERS, SMPLH_CONNECT, True, 'observation', (150/255, 190/255, 219/255))
        draw_sequence(predicted_positions, SMPLH_BODY_MEMBERS, SMPLH_CONNECT, True, 'prediction', (128/255, 197/255, 128/255))
        draw_sequence(positions[frame+obs:frame+obs+pred], SMPLH_BODY_MEMBERS, SMPLH_CONNECT, True, 'groundtruth', (150/255, 190/255, 219/255))