from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import random
import glob
import os

class BatchGeneratorAMASS(object):
    '''
    Batch Generator for trimmed AMASS dataset.
    '''
    def __init__(self,
                amass_data_path,
                amass_subset_name=None,
                sample_rate=4, 
                body_repr='joints' #['joints', 'marker_41', 'marker_67', 'smpl_params']
                ):
        '''
        Init function of BatchGeneratorAMASS.

        Args:
            - amass_data_path: path to amass_prodessed
            - amass_subset_name: subset(s) in AMASS, e.g. ACCAD
            - sample_rate: down sample rate
            - body_repr: body representation ['joints', 'marker_41', 'marker_67', 'smpl_params']

        Return: 
            - BatchGeneratorAMASS object
        '''
        self.rec_list = list()
        self.index_rec = 0
        self.amass_data_path = amass_data_path
        self.amass_subset_name = amass_subset_name
        self.sample_rate = sample_rate
        self.data_list = []
        self.gender_list = []
        self.betas_list = []
        self.transl_list = []
        self.glorot_list = []
        self.poses_list = []
        self.body_repr = body_repr

    def reset(self):
        '''
        Reset after a single whole epoch, and the data will be shuffled.

        Args:
            - none

        Return: 
            - none
        '''
        self.index_rec = 0
        random.shuffle(self.data_list)
        if self.body_repr in ['joints', 'marker_41', 'marker_67']:
            randnum = random.randint(0,100)
            np.random.seed(randnum)
            np.random.shuffle(self.data_all)
            np.random.seed(randnum)
            np.random.shuffle(self.gender_all)
            np.random.seed(randnum)
            np.random.shuffle(self.betas_all)
            np.random.seed(randnum)
            np.random.shuffle(self.transl_all)
            np.random.seed(randnum)
            np.random.shuffle(self.glorot_all)
            np.random.seed(randnum)
            np.random.shuffle(self.poses_all)

    def has_next_rec(self):
        '''
        Determine if there is a next batch.

        Args:
            - none

        Return: 
            - none
        '''
        if self.index_rec < len(self.data_list):
            return True
        return False

    def prepare_data(self, shuffle_seed=None):
        '''
        Init function of preparation.

        Args:
            - shuffle_seed: repeat results

        Return: 
            - none
        '''
        if self.amass_subset_name is not None:
            ## read the sequence in the subsets
            self.rec_list = []
            for subset in self.amass_subset_name:
                self.rec_list += glob.glob(os.path.join(self.amass_data_path,
                                                       subset,
                                                       '*.npz'  ))
        else:
            ## read all amass sequences
            self.rec_list = glob.glob(os.path.join(self.amass_data_path,
                                                    '*/*.npz'))

        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.rec_list)
        else:
            random.shuffle(self.rec_list) # shuffle recordings, not frames in a recording.

        
        print('[INFO] read all data to RAM...')
        for rec in self.rec_list:
            body_feature = {}
            if self.body_repr == 'smpl_params':
                pose = np.load(rec)['poses'][::self.sample_rate,:66] # 156d = 66d+hand
                transl = np.load(rec)['trans'][::self.sample_rate]
                if np.isnan(pose).any() or np.isinf(pose).any() or np.isnan(transl).any() or np.isinf(transl).any():
                    continue
                body_feature['transl'] = transl
                body_feature['pose'] = pose
            elif self.body_repr == 'joints':
                body_feature = np.load(rec)['joints'][::self.sample_rate].reshape([-1,22*3])
            elif self.body_repr == 'marker_41':
                body_feature = np.load(rec)['marker_cmu_41'][::self.sample_rate].reshape([-1,41*3])
            elif self.body_repr == 'marker_67':
                body_feature = np.load(rec)['marker_ssm2_67'][::self.sample_rate].reshape([-1,67*3])
            else:
                raise NameError('[ERROR] not valid body representation. Terminate')
            self.data_list.append(body_feature)
            self.gender_list.append(0 if np.load(rec)['gender'] == 'male' else 1)
            self.betas_list.append(np.load(rec)['betas'])
            self.transl_list.append(np.load(rec)['trans'][::self.sample_rate])
            self.glorot_list.append(np.load(rec)['poses'][::self.sample_rate, :3])
            self.poses_list.append(np.load(rec)['poses'][::self.sample_rate, 3:66])

        if self.body_repr in ['joints', 'marker_41', 'marker_67']:
            self.data_all = np.stack(self.data_list,axis=0)
            self.gender_all = np.stack(self.gender_list, axis=0)
            self.betas_all = np.stack(self.betas_list, axis=0)
            self.transl_all = np.stack(self.transl_list, axis=0)
            self.glorot_all = np.stack(self.glorot_list, axis=0)
            self.poses_all = np.stack(self.poses_list, axis=0)


    def next_batch_smplx_params(self, batch_size=64):
        '''
        Generate next batch in the form of body_parameters, different process.

        Args:
            - batch_size: batch size

        Return: 
            - [batch_pose, batch_transl]
        '''
        batch_pose_ = []
        batch_transl_ = []
        ii = 0
        while ii < batch_size:
            if not self.has_next_rec():
                break
            data = self.data_list[self.index_rec]
            batch_tensor_pose = torch.FloatTensor(data['pose']).unsqueeze(0)
            batch_tensor_transl = torch.FloatTensor(data['transl']).unsqueeze(0) #[b,t,d]
            batch_pose_.append(batch_tensor_pose)
            batch_transl_.append(batch_tensor_transl)
            ii = ii+1
            self.index_rec+=1
        batch_pose = torch.cat(batch_pose_,dim=0).permute(1,0,2) #[t,b,d]
        batch_transl = torch.cat(batch_transl_,dim=0).permute(1,0,2) #[t,b,d]
        return [batch_pose, batch_transl]


    def next_batch_kps(self, batch_size=64):
        '''
        Generate next batch in the form of 'markers'[joints, marker_41, marker_67].

        Args:
            - batch_size: batch size

        Return: 
            - batch_data: 'marker' data
        '''
        batch_data_ = self.data_all[self.index_rec:self.index_rec+batch_size]
        batch_gender_ = self.gender_all[self.index_rec:self.index_rec+batch_size]
        batch_betas_ = self.betas_all[self.index_rec:self.index_rec+batch_size]
        batch_transl_ = self.transl_all[self.index_rec:self.index_rec+batch_size]
        batch_glorot_ = self.glorot_all[self.index_rec:self.index_rec+batch_size]
        batch_poses_ = self.poses_all[self.index_rec:self.index_rec+batch_size]
        self.index_rec+=batch_size
        batch_data = torch.FloatTensor(batch_data_)
        batch_gender = torch.FloatTensor(batch_gender_)
        batch_betas = torch.FloatTensor(batch_betas_)
        batch_trnasl = torch.FloatTensor(batch_transl_)
        batch_glorot = torch.FloatTensor(batch_glorot_)
        batch_poses = torch.FloatTensor(batch_poses_)
        return batch_data, batch_gender, batch_poses, batch_betas, batch_trnasl, batch_glorot


    def next_batch(self, batch_size=64):
        '''
        The main funtion to generate batch.

        Args:
            - batch_size: batch size

        Return: 
            - batch_data: batch data
        '''
        if self.body_repr == 'smpl_params':
            batch = self.next_batch_smplx_params(batch_size)
        else:
            batch = self.next_batch_kps(batch_size)
        return batch


    def next_sequence(self):
        '''
        This function is only for produce files for visualization or testing in some cases
        compared to next_batch with batch_size=1, this function also outputs metainfo, like gender, body shape, etc.
        
        Args:
            - none

        Return: 
            - output: 1 batch data
        '''
        rec = self.rec_list[self.index_rec]

        body_feature = {}
        if self.body_repr == 'smpl_params':
            pose = np.load(rec)['poses'][::self.sample_rate,:66] # 156d = 66d+hand
            transl = np.load(rec)['trans'][::self.sample_rate]
            if np.isnan(pose).any() or np.isinf(pose).any() or np.isnan(transl).any() or np.isinf(transl).any():
                return None
            body_feature['transl'] = transl
            body_feature['pose'] = pose
        elif self.body_repr == 'joints':
            body_feature = np.load(rec)['joints'][::self.sample_rate].reshape([-1,22*3])
        elif self.body_repr == 'marker_41':
            body_feature = np.load(rec)['marker_cmu_41'][::self.sample_rate].reshape([-1,41*3])
        elif self.body_repr == 'marker_67':
            body_feature = np.load(rec)['marker_ssm2_67'][::self.sample_rate].reshape([-1,67*3])
        else:
            raise NameError('[ERROR] not valid body representation. Terminate')

        ## pack output data
        output = {}
        output['betas'] = np.load(rec)['betas'][:10]
        output['gender'] = np.load(rec)['gender']
        output['transl'] = transl
        output['glorot'] = pose[:,:3]
        output['poses'] = pose[:,3:]
        output['body_feature'] = body_feature
        output['transf_rotmat'] = np.load(rec)['transf_rotmat']
        output['transf_transl'] = np.load(rec)['transf_transl']

        self.index_rec += 1
        return output

    def get_feature_dim(self):
        '''
        Get dimension of body_representation feature.
        
        Args:
            - none

        Return: 
            - dim
        '''
        if self.body_repr == 'smpl_params':
            raise NameError('return list. No dim')
        elif self.body_repr == 'marker_41':
            return 41*3
        elif self.body_repr == 'marker_67':
            return 67*3
        elif self.body_repr == 'joints':
            return 22*3
        else:
            raise NameError('not implemented')


    def get_all_data(self):
        '''
        Get the data for a whole epoch.
        
        Args:
            - none

        Return: 
            - all data
        '''
        return torch.FloatTensor(self.data_all).permute(1,0,2) #[t,b,d]