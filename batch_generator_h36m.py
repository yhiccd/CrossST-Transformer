from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import random
import glob
import os

class BatchGeneratorH36M(object):
    '''
    Batch Generator for trimmed H36M dataset.
    '''
    def __init__(self,
                h36m_data_path,
                h36m_subset_name=None,
                actions=None,
                sample_rate=2,
                window=100
                # body_repr = 3d joints positions
                ):
        '''
        Init function of BatchGeneratorH36M.

        Args:
            - h36m_data_path: path to amass_prodessed
            - h36m_subset_name: subset(s) in H36M, e.g. "train 1, 6, 7, 8, 9, 11 test 5" or other sets
            - actions: walking etc.
            - sample_rate: down sample rate

        Return: 
            - BatchGeneratorAMASS object
        '''
        self.rec_list = list()
        self.index_rec = 0
        self.h36m_data_path = h36m_data_path
        self.h36m_subset_name = h36m_subset_name
        self.actions = actions
        self.sample_rate = sample_rate
        self.window = window
        self.data_list = []

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
        randnum = random.randint(0,100)
        np.random.seed(randnum)
        np.random.shuffle(self.data_all)

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
        if self.h36m_subset_name is not None:
            ## read the sequence in the subsets
            self.rec_list = []
            for subset in self.h36m_subset_name:
                self.rec_list += glob.glob(os.path.join(self.h36m_data_path,
                                                       'S{}'.format(subset),
                                                       '*.npz'  ))
        else:
            ## read all amass sequences
            self.rec_list = glob.glob(os.path.join(self.h36m_data_path,
                                                    '*/*.npz'))

        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.rec_list)
        else:
            random.shuffle(self.rec_list) # shuffle recordings, not frames in a recording.

        
        print('[INFO] read all data to RAM...')
        for rec in self.rec_list:
            body_feature = np.load(rec)['positions'][::self.sample_rate].reshape([-1,19*3])
            # optional: align h36m with amass
            # body_feature = body_feature / constant ## 2.0588

            # split data in to pieces
            # skip too short sequences
            n_frames = body_feature.shape[0]
            if n_frames < self.window:
                continue

            t = 0
            while t < n_frames:
                # get subsequence
                body_feature_piece = body_feature[t : t + self.window, :]

                # drop last, break if remaining frames are not sufficient
                if body_feature_piece.shape[0] < self.window:
                    break

                self.data_list.append(body_feature[t : t + self.window])
                t = t+self.window
            
                
        self.data_all = np.stack(self.data_list, axis=0)


    def next_batch_kps(self, batch_size=64):
        '''
        Generate next batch in the form of 3d joints positions

        Args:
            - batch_size: batch size

        Return: 
            - batch_data: 3d joints data
        '''
        batch_data_ = self.data_all[self.index_rec:self.index_rec+batch_size]
        batch_data = torch.FloatTensor(batch_data_)
        return batch_data, torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)


    def next_batch(self, batch_size=64):
        '''
        The main funtion to generate batch.

        Args:
            - batch_size: batch size

        Return: 
            - batch_data: batch data
        '''
        batch = self.next_batch_kps(batch_size)
        self.index_rec += batch_size
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

        body_feature = np.load(rec)['positions'][::self.sample_rate].reshape([-1,19*3])
       

        ## pack output data
        output = {}
        output['betas'] = 0
        output['gender'] = 0
        output['transl'] = 0
        output['glorot'] = 0
        output['poses'] = 0
        output['body_feature'] = body_feature
        output['transf_rotmat'] = 0
        output['transf_transl'] = 0

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
        
        return 19*3


    def get_all_data(self):
        '''
        Get the data for a whole epoch.
        
        Args:
            - none

        Return: 
            - all data
        '''
        return torch.FloatTensor(self.data_all).permute(1,0,2) #[t,b,d]