from typing import Union
import numpy as np
import torch
from colour import Color
from human_body_prior.body_model.body_model import BodyModel
from torch import nn
from human_body_prior.models.ik_engine import IK_Engine
from os import path as osp

class SourceKeyPoints(nn.Module):
    def __init__(self,
                 bm: Union[str, BodyModel],
                 n_joints: int=22,
                 kpts_colors: Union[np.ndarray, None] = None ,
                 ):
        super(SourceKeyPoints, self).__init__()

        self.bm = BodyModel(bm, persistant_buffer=False) if isinstance(bm, str) else bm
        self.bm_f = [] # self.bm.f
        self.n_joints = n_joints
        self.kpts_colors = np.array([Color('grey').rgb for _ in range(n_joints)]) if kpts_colors == None else kpts_colors

    def forward(self, body_parms):
        new_body = self.bm(**body_parms)


        return {'source_kpts':new_body.Jtr[:,:self.n_joints], 'body': new_body}

class IK_Fitting:
    def __init__(self, 
                 support_dir='./support_data/body_models', 
                 vposer_expr_dir='V02_05', 
                 bm_fname_neutral='smplx_hpp/neutral/model.npz', 
                 bm_fname_male='smplx_hpp/male/model.npz', 
                 bm_fname_female='smplx_hpp/female/model.npz', 
                 n_joints=22, 
                 data_loss='L2', 
                 stepwise_weights=[{'data': 10., 'poZ_body': .01, 'betas': .5},], 
                 optimizer_args={'type':'ADAM', 'max_iter':200, 'lr':1e-1, 'tolerance_change': 1e-4},
                 device=torch.device('cuda', index=4)) -> None:
        self.support_dir = support_dir # './support_data/body_models'
        self.vposer_expr_dir = osp.join(self.support_dir, vposer_expr_dir) #in this directory the trained model along with the model code exist
        self.bm_fname_neutral =  osp.join(self.support_dir, bm_fname_neutral) #'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
        self.bm_fname_male =  osp.join(self.support_dir, bm_fname_male) # 'smplx_hpp/male/model.npz'
        self.bm_fname_female =  osp.join(self.support_dir, bm_fname_female) # 'smplx_hpp/female/model.npz'
        self.bm_fname = None
        self.comp_device = device
        self.n_joints = n_joints # 22
        red = Color("red")
        blue = Color("blue")
        self.kpts_colors = [c.rgb for c in list(red.range_to(blue, self.n_joints))]
        # create source and target key points and make sure they are index aligned
        if data_loss == 'L1':
            self.data_loss = torch.nn.L1Loss(reduction='sum')
        elif data_loss == 'L2':
            self.data_loss = torch.nn.MSELoss(reduction='sum')
        else:
            raise ValueError('Please use L1 loss or L2 loss, other loss functions need to be implemented by modifying the source code.')
        self.stepwise_weights = stepwise_weights # [{'data': 10., 'poZ_body': .01, 'betas': .5},]
        # self.optimizer_args = {'type':'LBFGS', 'max_iter':300, 'lr':1, 'tolerance_change': 4e-3, 'history_size':200}
        self.optimizer_args = optimizer_args # {'type':'ADAM', 'max_iter':200, 'lr':1e-1, 'tolerance_change': 1e-4}
        self.ik_engine = IK_Engine(vposer_expr_dir=self.vposer_expr_dir,
                            verbosity=1,
                            display_rc= (2, 2),
                            data_loss=self.data_loss,
                            stepwise_weights=self.stepwise_weights,
                            optimizer_args=self.optimizer_args,
                            device=device).to(self.comp_device)

    def fittingOP(self, 
                  target, 
                  gender, 
                  pose, 
                  beta, 
                  transl, 
                  glorot):
        root_orient = glorot
        pose_body = pose
        trans = transl
        betas = beta[:, :10]
        # for batch data
        batch_size = target.shape[0]
        frame_size = target.shape[1]
        #---------------------------Code Reconstruction--------------------------
        # paras = []
        # joints = []
        # for bz in range(batch_size):
        #     print('-------batch:{}-------'.format(bz))
            
        #     # test if neutral model does well too
        #     # self.bm_fname = self.bm_fname_male if gender[bz] == 0 else self.bm_fname_female
        #     self.bm_fname = self.bm_fname_neutral
            
        #     target_pts = target[bz].reshape(frame_size, self.n_joints, -1).detach().to(self.comp_device)
        #     source_pts = SourceKeyPoints(bm=self.bm_fname, n_joints=self.n_joints, kpts_colors=self.kpts_colors).to(self.comp_device)
        #     ik_res = self.ik_engine(source_pts, target_pts, {'betas': betas[bz].repeat(frame_size, 1).to(self.comp_device),
        #                                                      'pose_body': pose_body[bz].to(self.comp_device),
        #                                                      'trans': trans[bz].to(self.comp_device),
        #                                                      'root_orient': root_orient[bz].to(self.comp_device)})

        #     ik_res_detached = {k: v.detach() for k, v in ik_res.items()}
        #     paras.append(ik_res_detached)
        #     joints.append(source_pts(ik_res_detached)['source_kpts'].detach().clone())
        #     source_pts(ik_res_detached)
        #     nan_mask = torch.isnan(ik_res_detached['trans']).sum(-1) != 0
        #     if nan_mask.sum() != 0: raise ValueError('Sum results were NaN!')
        # output = {}
        # output['betas'] = torch.stack([paras[i]['betas'] for i in range(len(paras))], dim=0)
        # output['trans'] = torch.stack([paras[i]['trans'] for i in range(len(paras))], dim=0)
        # output['root_orient'] = torch.stack([paras[i]['root_orient'] for i in range(len(paras))], dim=0)
        # output['pose_body'] = torch.stack([paras[i]['pose_body'] for i in range(len(paras))], dim=0)
        #---------------------------Code Reconstruction--------------------------

        self.bm_fname = self.bm_fname_neutral
        target_pts = target.reshape(batch_size * frame_size, self.n_joints, -1).detach().to(self.comp_device)
        source_pts = SourceKeyPoints(bm=self.bm_fname, n_joints=self.n_joints, kpts_colors=self.kpts_colors).to(self.comp_device)
        ik_res = self.ik_engine(source_pts, target_pts, {'betas': torch.stack([betas[i].repeat(frame_size, 1) for i in range(batch_size)], dim=0).reshape(batch_size * frame_size, -1).to(self.comp_device),
                                                         'pose_body': pose_body.reshape(batch_size * frame_size, -1).to(self.comp_device),
                                                         'trans': trans.reshape(batch_size * frame_size, -1).to(self.comp_device),
                                                         'root_orient': root_orient.reshape(batch_size * frame_size, -1).to(self.comp_device)})
        ik_res_detached = {k: v.detach() for k, v in ik_res.items()}
        nan_mask = torch.isnan(ik_res_detached['trans']).sum(-1) != 0
        if nan_mask.sum() != 0: raise ValueError('Sum results were NaN!')

        return source_pts(ik_res_detached)['source_kpts'].reshape(batch_size, frame_size, self.n_joints, -1).detach().clone(), ik_res_detached