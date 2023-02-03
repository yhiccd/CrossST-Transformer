import os
from re import A
import sys
import datetime
import time

from matplotlib.pyplot import bone
from sympy import false

from batch_generator_h36m import BatchGeneratorH36M

# set root path
Root_Dir = os.path.abspath('') # set root path
sys.path.append(Root_Dir)

import argparse
from batch_generator_amass import BatchGeneratorAMASS
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from cross_transformer_model import get_model, create_look_ahead_mask
from utils import INTERVAL, METRIC_TARGET_LENGTHS_AMASS, METRIC_PCK_THRESHS, remove_files, save_model
from ik_class import IK_Fitting
from fk import H36M_CONNECT
from fk import SMPLH_CONNECT
from metrics import MetricsEngine
from fk import SMPLHForwardKinematics

train_subjects = [1, 6, 7, 8, 9, 11]
validate_subjects = [5]
actions = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]

def init_weights(module):
    '''
    init weights, maybe consider deep norm
    '''
    classname = module.__class__.__name__
    if classname.find('Linear') != -1:        
        torch.nn.init.kaiming_normal_(module.weight)

def skeleton_integrity_loss(prediction, ground_truth):
    '''
    Calc the skeleton integrity loss for a batch prediction

    Args:
        - prediction: the prediction (batch_size, seq_len, num_joints, rep)
        - ground_truth: in the shape of (batch_size, seq_len, num_joints, rep)
    
    Return: 
        - the output: integrity loss
    '''
    batch_size = prediction.shape[0]
    seq_len = prediction.shape[1]
    P_i = prediction
    P_j = ground_truth
    G_i = torch.matmul(P_i, P_i.transpose(-1, -2))
    G_j = torch.matmul(P_j, P_j.transpose(-1, -2))
    Singu_G = torch.matmul(P_j.transpose(-1, -2), P_i)
    u, s, v = torch.svd(Singu_G)
    loss_total = 0
    for batch in range(batch_size):
        for seq in range(seq_len):
            loss_total += G_i[batch][seq].trace() + G_j[batch][seq].trace() - 2*torch.sum(s[batch][seq])
    loss_total /= batch_size*seq_len
    return loss_total

def bone_length_loss(prediction, ground_truth, type):
    '''
    Calc the bone length loss for a batch prediction

    Args:
        - prediction: the prediction (batch_size, seq_len, num_joints, rep)
        - ground_truth: in the shape of (batch_size, seq_len, num_joints, rep)
        - type: amass or h36m
    
    Return: 
        - the output: bone length loss
    '''
    if type == 'amass':
        connect = SMPLH_CONNECT
    else:
        connect = H36M_CONNECT
    loss_total = 0
    for bone in connect:
        pred_bone_len = torch.norm(prediction[:, :, bone[0]] - prediction[:, :, bone[1]], 2, 2)
        gt_bone_len = torch.norm(ground_truth[:, :, bone[0]] - ground_truth[:, :, bone[1]], 2, 2)
        loss_total += torch.mean(abs(pred_bone_len - gt_bone_len))
    loss_total /= len(connect)
    return loss_total * 1000

def total_loss(prediction, ground_truth, regression, mode):
    SKI = 0
    JL = 0
    BLL = 0
    REG = 0
    if 'SKI' in mode:
        # loss SKI
        SKI = skeleton_integrity_loss(prediction, ground_truth)
    
    if 'JL' in mode:
        # loss JL
        JL = torch.nn.functional.mse_loss(prediction, ground_truth, reduction='sum')
    
    if 'REG' in mode:
        # loss REG
        if regression != None:
            REG = torch.nn.functional.mse_loss(prediction, regression, reduction='sum')
        else:
            REG = 0
    # in total
    loss_r = SKI * args.lambda_ski + JL * args.lambda_jl + REG * args.lambda_reg
    return loss_r, np.array([loss_r.item(), SKI.item(), JL.item(), 0 if REG == 0 else REG.item()])

def mpjpe_error(prediction, ground_truth):
    '''
    Calc the mpjpe_error

    Args:
        - prediction: the prediction (batch_size, seq_len, num_joints, rep)
        - ground_truth: in the shape of (batch_size, seq_len, num_joints, rep)
    
    Return: 
        - mpjpe: mpjpe error
    '''
    prediction = prediction.contiguous().view(-1, 3)  
    ground_truth = ground_truth.contiguous().view(-1, 3)
    mpjpe = torch.mean(torch.norm(ground_truth - prediction, 2, 1))    
    return mpjpe


def adjust_learning_rate(args, optimizer, global_step):
    '''
    Sets the learning rate to the initial LR decayed
    '''
    d_model = float(args.d_model)
    step = float(global_step)
    arg1 = 1/pow(step, 0.5)
    arg2 = step * (args.warm_up_steps ** -1.5)
    lr = 1/(d_model, 0.5) * min(arg1, arg2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None, gamma=0.1):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=gamma)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler

def train(epoch):
    '''
    train model
    
    Args:
        - epoch: epoch

    '''
    batch_start_time = time.time()
    train_losses = 0
    total_num_sample = 0
    # loss_names_nonreg = ['TOTAL', 'SKI', 'JL'] # without 'REG'
    # loss_names = ['TOTAL', 'SKI', 'JL', 'REG'] # 'REG'
    loss_names = args.loss_mode

    while batch_gen_train.has_next_rec():
        batch_data, batch_gender, batch_poses, batch_betas, batch_transl, batch_glorot = [i.to(device) for i in batch_gen_train.next_batch(args.batch_size)]
        # drop last
        if batch_data.shape[0] != args.batch_size:
            continue
        # no need to define gt first
        # ground_truth = batch_data[:, args.obs_seq_len:].reshape(args.batch_size, args.horizon, args.num_joints, -1)
        if (torch.isnan(batch_data)).any():
            print('-- meet nan input data. skip and continue')
            continue
        # pred_seqs = []
        
        for horizon in range(args.horizon):
            obs_seq = batch_data[:, horizon : horizon+args.obs_seq_len].reshape(args.batch_size, args.obs_seq_len, args.num_joints, -1)
            prediction, attention_weight = model(obs_seq, mask_obs)
            
            # new version considers all frames, i.e. input:(0-14) output:(1-15)' use gt(1-15) to calc loss
            regression, output = fitting_engine.fittingOP(prediction.detach().clone(),
                                                batch_gender,
                                                batch_poses[:, horizon + args.pred_seq_len : horizon + args.pred_seq_len + args.obs_seq_len],
                                                batch_betas,
                                                batch_transl[:, horizon + args.pred_seq_len : horizon + args.pred_seq_len + args.obs_seq_len],
                                                batch_glorot[:, horizon + args.pred_seq_len : horizon + args.pred_seq_len + args.obs_seq_len])
            loss, losses = total_loss(prediction, batch_data[:, horizon + args.pred_seq_len : horizon + args.pred_seq_len + args.obs_seq_len].reshape(args.batch_size, args.obs_seq_len, args.num_joints, -1), regression, loss_names)

            # print('MAE:{}'.format(torch.nn.functional.mse_loss(output, batch_poses[:, horizon + args.pred_seq_len : horizon + args.pred_seq_len + args.obs_seq_len], reduction='sum')))
            # whole frame 
            # loss = torch.nn.functional.mse_loss(prediction, batch_data[:, horizon + args.pred_seq_len : horizon + args.pred_seq_len + args.obs_seq_len].reshape(args.batch_size, args.obs_seq_len, args.num_joints, -1), reduction='sum')
            # single loss test
            # loss = torch.nn.functional.mse_loss(prediction[:, -1], batch_data[:, horizon + args.obs_seq_len + args.pred_seq_len - 1].reshape(args.batch_size, args.num_joints, -1), reduction='sum')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses += losses
            total_num_sample += 1
            print('it:{} loss:{}'.format(total_num_sample, loss))
            # print("[Epoch %d/%d] [Batch %d/%d] [L2 loss: %f] [Skeleton Integrity loss: %f] [time cost: %f s]"
            # % (epoch + 1, args.num_epochs, total_num_sample, args.num_epochs / args.batch_size, loss_l2.item(), loss_si.item(), time_tem))
    batch_gen_train.reset()
    scheduler.step()
    time_cost = time.time() - batch_start_time
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('Train ====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, time_cost, losses_str, lr))
    for name, loss in zip(loss_names, train_losses):
        writer.add_scalar(name, loss, epoch)
    
    if (epoch + 1) % 10 == 0:
        print('----saving model-----')
        save_model(model, model_save_dir, epoch)

def validate(epoch, interval):
    '''
    validate/evaluate model
    
    Args:
        - epoch: epoch

    '''
    batch_start_time = time.time()
    validate_error = []
    for i, inter in enumerate(interval):
        validate_error.append(0)
    total_num_sample = 0

    while batch_gen_validate.has_next_rec():
        batch_data, batch_gender, batch_poses, batch_betas, batch_transl, batch_glorot = [i.to(device) for i in batch_gen_validate.next_batch(args.batch_size)]
        # drop last
        if batch_data.shape[0] != args.batch_size:
            continue

        if (torch.isnan(batch_data)).any():
            print('-- meet nan input data. skip and continue')
            continue

        obs_seq = batch_data[:, : args.obs_seq_len].reshape(args.batch_size, args.obs_seq_len, args.num_joints, -1)
        ground_truth_seq = batch_data[:, args.obs_seq_len : ].reshape(args.batch_size, args.horizon, args.num_joints, -1)
        forecast_seq = inference(model, obs_seq, args.obs_seq_len, args.pred_seq_len, args.horizon)
        for i, inter in enumerate(interval):
            validate_error[i] += mpjpe_error(forecast_seq[:, : inter], ground_truth_seq[:, : inter]) * 1000
        total_num_sample += 1
        
    batch_gen_validate.reset()
    time_cost = time.time() - batch_start_time
    for i, inter in enumerate(interval):
        validate_error[i] /= total_num_sample
    losses_str = 'MPJPE:' + ' '.join(['{}: {:.4f}'.format(inter, error) for inter, error in zip(INTERVAL, validate_error)])
    logger.info('Validate ====> Epoch: {} Time: {:.2f} {}'.format(epoch, time_cost, losses_str))
    for name, loss in zip(INTERVAL, validate_error):
        writer.add_scalar("MPJPE{}".format(name), loss, epoch)

    # metrics_engine.compute_and_aggregate(inference_seq, ground_truth_seq)
    # final_metrics = metrics_engine.get_final_metrics()
    # feed_dict, record_flag = metrics_engine.get_summary_feed_dict(final_metrics)
    # metrics_engine.reset()

    # for item in feed_dict.keys(): 
    #     # name = "{}/until_{}".format(item, t)
    #     name = str(item)
    #     writer.add_scalar(name, feed_dict[item].item(), epoch + 1)
    
    # if record_flag:
    #     if not os.path.exists(model_save_dir):
    #         os.makedirs(model_save_dir)
    #     remove_files(model_save_dir, '*.pt', True)
    #     save_model(model, None, model_save_dir, epoch + 1)
    # # without GAN structure
    # mask_obs = create_look_ahead_mask(args.obs_seq_len).cuda()
    
    # prediction, attention_weight = model(obs_seq, mask_obs)
    # loss_l2 = l2_loss(prediction, ground_truth_seq_once[:, args.pred_seq_len:, :, :])
    # loss_si = skeleton_integrity_loss(prediction, ground_truth_seq_once[:, args.pred_seq_len:, :, :])
    
    # writer.add_scalar('Test/L2 loss without GAN', loss_l2.item(), epoch + 1)
    # writer.add_scalar('Test/Skeleton Integrity loss without GAN', loss_si.item(), epoch + 1)
    # batch_end_time = time.time()
    # time_tem = batch_end_time - batch_start_time
    # print("[Test] [Iters %d] [L2 loss: %f] [Skeleton Integrity loss: %f] [time cost: %f s]"
    #         % (epoch + 1, loss_l2.item(), loss_si.item(), time_tem))

def inference(model, obs_seq, obs_seq_len, pred_seq_len, horizon):
    '''
    inference future frames based on obs sequence

    Args:
        - model: generator(with GAN) or model (without GAN)
        - obs_seq: obs sequence(seed sequence)
        - obs_seq_len: length of obs sequence
        - pred_seq_len: length of model's output prediction
        - horizon: total length of sequence to be inferenced
    
    Returns:
        - forecast_steps: The inferred prediction sequence

    '''
    model.eval()
    with torch.no_grad():
        step = 0
        forecast_steps = torch.zeros(obs_seq.shape[0], horizon, obs_seq.shape[2], obs_seq.shape[3]).cuda()
        mask = create_look_ahead_mask(obs_seq_len).cuda()
        while step < horizon:
            # forecast_result in the shape of (batch_size, pred_seq_len, num_joints, rep)
            forecast_result, attention = model(obs_seq, mask)
            obs_seq[:, :obs_seq_len - pred_seq_len, :, :] = obs_seq[:, pred_seq_len:, :, :].clone()
            obs_seq[:, obs_seq_len - pred_seq_len:, :, :] = forecast_result[:, -pred_seq_len:, :, :].clone()
            forecast_steps[:, step:min(horizon - step, pred_seq_len) + step, :, :] = \
                forecast_result[:, obs_seq_len-pred_seq_len:min(horizon - step, pred_seq_len) + obs_seq_len-pred_seq_len, :, :].detach()
            step += min(horizon - step, pred_seq_len)
    return forecast_steps

if __name__ == '__main__':
    # record start time
    start_time = time.time()
    # default experiment ID
    time_now = datetime.datetime.now()
    default_experiment_id = str(time_now)

    parser = argparse.ArgumentParser()

    # Infrastructure specific parameters
    parser.add_argument('--seed', type=int, default=42, help='Seed Value')
    parser.add_argument('--experiment_id', type=str, default=default_experiment_id, help='Unique experiment id to restore an existing model.')
    parser.add_argument('--tensorboard', type=str, default='logs/tb/', help='log_dir')
    parser.add_argument('--save_dir', default='models/', type=str, help='Path to save models.')
    parser.add_argument('--from_config', type=str, help='Path to an existing config.json to start a new experiment.')
    parser.add_argument('--print_frequency', type=int, default=100, help='Print/log every \'print_frequency\' training steps.')
    parser.add_argument('--test_frequency', type=int, default=1000, help='Runs validation every \'test_frequency\' training steps.')

    # Data specific parameters
    parser.add_argument('--dataset_mode', type=str, default='amass', choices=['amass', 'h36m'], help='which type of dataset.')
    parser.add_argument('--data_dir', type=str, default="./support_data/amass_processed", help='Path to data.')
    parser.add_argument('--train_sub_dataset', nargs='+', default=['CMU', 'MPI_HDM05'], help='train dataset to be used.')
    parser.add_argument('--test_sub_dataset', nargs='+', default=['ACCAD', 'BMLhandball'], help='test dataset to be used.')
    parser.add_argument('--validate_sub_dataset', nargs='+', default=['HumanEva'], help='validate dataset to be used.')
    parser.add_argument('--sample_rate', type=int, default=4, help='Down sample rate.') # 8
    parser.add_argument('--body_repr', type=str, default='joints', help='Body representation.')

    # IK_engine specific parameters
    parser.add_argument('--support_dir', type=str, default='./support_data/body_models', help='Path to body models.')
    parser.add_argument('--vposer_expr_dir', type=str, default='V02_05', help='Path to VPoser.')
    parser.add_argument('--bm_fname_neutral', type=str, default='smplx_hpp/neutral/model.npz', help='Path to smplx_neutral.')
    parser.add_argument('--bm_fname_male', type=str, default='smplx_hpp/male/model.npz', help='Path to smplx_male.')
    parser.add_argument('--bm_fname_female', type=str, default='smplx_hpp/female/model.npz', help='Path to smplx_female.')
    parser.add_argument('--num_joints', type=int, default=22, help='number of joints.')
    parser.add_argument('--data_loss', type=str, default='L2', help='data loss, L1 or L2 for now.')
    parser.add_argument('--stepwise_weights', type=list, default=[{'data': 10., 'poZ_body': .01, 'betas': .5},], help='weights for fitting loss.')
    parser.add_argument('--optimizer_args', type=dict, default={'type':'ADAM', 'max_iter':300, 'lr':1e-1, 'tolerance_change': 1e-4}, help='optimizer config.')

    # Model specific parameters
    parser.add_argument('--residual_velocity', type=bool, default=True, help='Add a residual connection that effectively models velocities.')
    parser.add_argument('--obs_seq_len', type=int, default=25, help='Length of frames to feed into the model.') # 15
    parser.add_argument('--pred_seq_len', type=int, default=1, help='Offset frames of prediction.')
    parser.add_argument('--horizon', type=int, default=75, help='total length inference sequence.') # 45
    parser.add_argument('--shared_qkv', type=bool, default=True, help='shared weights for making Q, K and V.')
    parser.add_argument('--d_model', type=int, default=64, help='Size of d_model of the transformer')
    parser.add_argument('--dq', type=int, default=128, help='Size of d_q of the transformer')
    parser.add_argument('--dk', type=int, default=128, help='Size of d_k of the transformer')
    parser.add_argument('--dv', type=int, default=64, help='Size of d_v of the transformer')
    parser.add_argument('--num_layers', nargs='+', default=[[1, 1, 1], [1, 3, 1]], help='Number of layers of the transformer')
    parser.add_argument('--num_heads_temporal', type=int, default=8, help='Number of heads of the transformer\'s temporal block')
    parser.add_argument('--num_heads_spatial', type=int, default=8, help='Number of heads of the transformer\'s spatial block')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Dropout rate.')

    # Optimization specific parameters
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning Rate')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.98, help='Learning rate multiplier. See torch.optim.lr_scheduler.StepLR.')
    parser.add_argument('--adam_b1', type=float, default=0.9, help='adam_bates1.')
    parser.add_argument('--adam_b2', type=float, default=0.999, help='adam_bates2.')
    parser.add_argument('--learning_rate_decay_steps', type=int, default=100, help='Decay steps. See exponential_decay & torch.optim.lr_scheduler.StepLR.')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimization function type.')
    parser.add_argument('--policy', type=str, default='lambda', choices=['lambda', 'step', 'plateau'], help='Policy of LR change.')
    parser.add_argument('--lambda_ski', type=float, default=0.2, help='ski weight.')
    parser.add_argument('--lambda_jl', type=float, default=0.5, help='jl weight.')
    parser.add_argument('--lambda_reg', type=float, default=0.3, help='reg weight.')

    # Training specific parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to use during training.')
    parser.add_argument('--num_epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--warm_up_steps', type=int, default=1000, help='warm_up steps for LR')
    parser.add_argument('--early_stopping_tolerance', type=int, default=20, help='of waiting steps until the validation loss improves.')
    parser.add_argument('--gpuid', type=int, default=7, help='GPU ID')
    parser.add_argument('--loss_mode', nargs='+', default=['TOTAL', 'SKI', 'JL'], help='loss mode')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_dir', type=str)
    args = parser.parse_args()
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7,0"

    # -----------
    # create dirs
    # -----------
    if args.resume:
        if os.path.isfile(args.resume_dir + '/checkpoint'):
            checkpoint = torch.load(args.resume_dir + '/checkpoint')
            record_dir = checkpoint['record_dir']
            model_dir = checkpoint['model_dir']
            result_dir = checkpoint['result_dir']
            log_dir = checkpoint['log_dir']
            tb_dir = checkpoint['tb_dir']
    else:
        record_dir = '%s/%s' % ('experiments', args.experiment_id)
        model_dir = '%s/models' % record_dir
        result_dir = '%s/results' % record_dir
        log_dir = '%s/log' % record_dir
        tb_dir = '%s/tb' % record_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)

    # ------------
    # Logging prep
    # ------------
    logger = create_logger(os.path.join(log_dir, 'log.txt'))
    logger.info("logger init successfully")
    device = torch.device('cuda', index=args.gpuid) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)
    logger.info("On device : {}".format(args.gpuid))
    writer = SummaryWriter(log_dir=tb_dir, comment=str(args.obs_seq_len) + "-" + str(args.pred_seq_len))
    logger.info("TensorBoard writter initiating")
    model_save_dir = model_dir
    logger.info("Model save dir : {}".format(model_save_dir))
    config = ''
    for k,v in sorted(vars(args).items()):
        config += '\n' + k + ' : ' + str(v)
    logger.info("Config of this experiment : {}".format(config))

    # ------------
    # Dataset prep
    # ------------
    if args.dataset_mode == 'amass':
        batch_gen_train = BatchGeneratorAMASS(args.data_dir,
                                            args.train_sub_dataset,
                                            args.sample_rate,
                                            args.body_repr)
        batch_gen_train.prepare_data()

        batch_gen_validate = BatchGeneratorAMASS(args.data_dir,
                                                args.validate_sub_dataset,
                                                args.sample_rate,
                                                args.body_repr)
        batch_gen_validate.prepare_data()
    else :
        batch_gen_train = BatchGeneratorH36M(h36m_data_path=args.data_dir, 
                                                 h36m_subset_name=train_subjects,
                                                 sample_rate=2,
                                                 actions=actions)
        batch_gen_train.prepare_data()

        batch_gen_validate = BatchGeneratorH36M(h36m_data_path=args.data_dir,
                                                 h36m_subset_name=validate_subjects,
                                                 sample_rate=2,
                                                 actions=actions)
        batch_gen_validate.prepare_data()
    if args.dataset_mode == 'amass':
        logger.info("Dataset initiating\n{}\n{}\n{}".format('train: '+str(args.train_sub_dataset),
                                                            'test: '+str(args.test_sub_dataset),
                                                            'validate: '+str(args.validate_sub_dataset)))
    else :
        logger.info("Dataset initiating\n{}\n{}\n".format('train: H36M'+str(train_subjects),
                                                            'validate: H36M'+str(validate_subjects)))
    # ----------
    # Model prep
    # ----------
    model = get_model(args)
    model.type(torch.cuda.FloatTensor).train()
    model.to(device)
    logger.info('Model initiating : \n{}'.format(model))

    # ---------------
    # IK_Fitting prep
    # ---------------
    fitting_engine = IK_Fitting(support_dir=args.support_dir,
                            vposer_expr_dir=args.vposer_expr_dir,
                            bm_fname_neutral=args.bm_fname_neutral,
                            bm_fname_male=args.bm_fname_male,
                            bm_fname_female=args.bm_fname_female,
                            n_joints=args.num_joints,
                            data_loss=args.data_loss,
                            stepwise_weights=args.stepwise_weights,
                            optimizer_args=args.optimizer_args,
                            device=torch.device('cuda', index=args.gpuid))
    logger.info('Fitting engine initiating')

    # ---------------
    # Optimizers prep
    # ---------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.adam_b1, args.adam_b2))
    logger.info('Optimizer initiating, use {}'.format(args.optimizer))
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=100, nepoch=500)
    # scheduler = get_scheduler(optimizer, policy='step', decay_step=args.learning_rate_decay_steps, gamma=args.learning_rate_decay_rate)
    logger.info('Scheduler initiating, use {}'.format(args.policy))
    
    # ---------------
    # Resume prep
    # --------------- 
    epoch = 0
    if args.resume:
        epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Loss function
    # Metrics engine in metrics.py
    l2_loss = torch.nn.MSELoss()
    fk_engine = SMPLHForwardKinematics()

    # ----------
    #  Training
    # ---------- 
    # torch.autograd.set_detect_anomaly(True)s
    mask_obs = create_look_ahead_mask(args.obs_seq_len).to(device)
    # for epoch in range(args.num_epochs):
    while epoch < args.num_epochs:
        train(epoch)
        if (epoch + 1) % 10 == 0:
            validate(epoch, INTERVAL)
        checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'record_dir': record_dir,
        'model_dir': model_dir,
        'result_dir': result_dir,
        'log_dir': log_dir,
        'tb_dir':tb_dir,
        }

        torch.save(checkpoint, record_dir + '/checkpoint')
        epoch += 1
    writer.close()
    end_time = time.time()
    print('time cost',end_time-start_time,'s')
