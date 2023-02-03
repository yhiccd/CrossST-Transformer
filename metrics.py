import numpy as np
import cv2
import quaternion
import copy
import torch

from utils import is_valid_rotmat, rotmat2euler, aa2rotmat
from utils import get_closest_rotmat, sparse_to_full, local_rot_to_global


def positional(predictions, targets):
    '''
    Computes the Euclidean distance between joints in 3D space.
    3 in (..., n_joints, 3) represents the 3D coordinates for each joint
    
    Args:
        - predictions: tensor of predicted 3D joint positions in format (..., n_joints, 3)
        - targets: ground truth tensor of same shape as `predictions`

    Returns:
        The Euclidean distance for each joint as an tensor of shape (..., n_joints)
    '''
    #  / num_joints MPJPE
    return torch.sqrt(torch.sum((predictions - targets) ** 2, dim=-1))


def pck(predictions, targets, thresh):
    '''
    Percentage of correct keypoints.
    predictions & targets should be in the 3d position format
    
    Args:
        - predictions: tensor of predicted 3D joint positions in format (..., n_joints, 3)
        - targets: tensor of same shape as `predictions`
        - thresh: radius within which a predicted joint has to lie.

    Returns:
        Percentage of correct keypoints at the given threshold level, stored in a tensor of shape (..., len(threshs))

    '''
    # distance for each joint
    dist = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=-1))
    pck = torch.mean(torch.tensor((dist <= thresh).clone(), dtype = torch.float32), axis =-1)
    return pck


def angle_diff(predictions, targets):
    '''
    Computes the angular distance between the target and predicted rotations. We define this as the angle that is
    required to rotate one rotation into the other. 
    This essentially computes || log(R_diff) || , where R_diff is the
    difference rotation between prediction and target.
    predictions & targets in the rotmat format

    Args:
        - predictions: tensor of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        - targets: tensor of same shape as `predictions`

    Returns:
        The geodesic distance for each joint as an tensor of shape (..., n_joints)
    '''
    assert predictions.shape[-1] == predictions.shape[-2] == 3
    assert targets.shape[-1] == targets.shape[-2] == 3

    ori_shape = predictions.shape[:-2]
    preds = predictions.reshape(-1, 3, 3)
    targs = targets.reshape(-1, 3, 3)

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = torch.matmul(preds, targs.permute(0, 2, 1))

    # convert `r` to angle-axis representation and extract the angle, which is our measure of difference between
    # the predicted and target orientations
    angles = []
    for i in range(r.shape[0]):
        aa, _ = cv2.Rodrigues(r[i].cpu().numpy()) # r[i] in the shape of 3*3
        angles.append(torch.linalg.norm(torch.from_numpy(aa).cuda()))
    angles = torch.tensor(angles)
    return angles.reshape(ori_shape)


def euler_diff(predictions, targets):
    '''
    Computes the Euler angle error as in previous work, following
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L207
    
    Args:
        - predictions: tensor of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        - targets: tensor of same shape as `predictions`

    Returns:
        The Euler angle error an tensor of shape (..., )
    '''
    assert predictions.shape[-1] == 3 and predictions.shape[-2] == 3
    assert targets.shape[-1] == 3 and targets.shape[-2] == 3
    n_joints = predictions.shape[-3]

    ori_shape = predictions.shape[:-3]
    preds = predictions.reshape(-1, 3, 3)
    targs = targets.reshape(-1, 3, 3)

    euler_preds = rotmat2euler(preds)  # (N, 3)
    euler_targs = rotmat2euler(targs)  # (N, 3)

    # reshape to (-1, n_joints*3) to be consistent with previous step
    euler_preds = euler_preds.reshape(-1, n_joints*3)
    euler_targs = euler_targs.reshape(-1, n_joints*3)

    # l2 error on euler angles, use standard error
    idx_to_use = torch.where(torch.std(euler_targs, dim = 0) > 1e-4)[0]
    euc_error = torch.pow(euler_targs[:, idx_to_use] - euler_preds[:, idx_to_use], 2)
    euc_error = torch.sqrt(torch.sum(euc_error, dim=1))  # (-1, ...)

    # reshape to original
    # return euc_error.reshape(ori_shape)
    return euc_error.reshape(ori_shape)


class MetricsEngine(object):
    '''
    Compute and aggregate various motion metrics. 
    It keeps track of the metric values "per frame", 
    so that we can evaluate them for different sequence lengths.
    '''
    def __init__(self, fk_engine, target_lengths, force_valid_rot, rep, which=None, pck_threshs=None, is_sparse=True):
        '''
        Initializer.
        
        Args:
            - fk_engine: An object of type `ForwardKinematics` used to compute positions.
            - target_lengths: List of target sequence lengths that should be evaluated.
            - force_valid_rot: If True, the input rotation matrices might not be valid rotations and so it will find
              the closest rotation before computing the metrics.
            - rep: Which representation format to use, "quat" or "rotmat".
            - which: Which metrics to compute. Options are [positional, joint_angle, pck, euler], defaults to all.
            - pck_threshs: List of thresholds for PCK evaluations.
            - is_sparse:  If True, `n_joints` is assumed to be 15, otherwise the full SMPL skeleton is assumed. If it is
              sparse, the metrics are only calculated on the given joints.
        '''
        self.which = which if which is not None else ["positional", "joint_angle", "pck", "euler"]
        self.target_lengths = target_lengths
        self.force_valid_rot = force_valid_rot
        self.fk_engine = fk_engine
        self.pck_threshs = pck_threshs if pck_threshs is not None else [0.2]
        self.is_sparse = is_sparse
        self.all_summaries_op = None
        self.n_samples = 0
        self._should_call_reset = False  # a guard to avoid stupid mistakes
        self.rep = rep
        assert self.rep in ["rotmat", "quat", "aa"]
        # assert is_sparse, "at the moment we expect sparse input; if that changes, " \
                        #   "the metrics values may not be comparable anymore"

        # treat pck_t as a separate metric
        if "pck" in self.which:
            self.which.pop(self.which.index("pck"))
            for t in self.pck_threshs:
                self.which.append("pck_{}".format(int(t*100) if t*100 >= 1 else t*100))
        
        # make a aggregation dict self.metrics_agg
        # {'positional': None, 'joint_angle': None, 'euler': None, 'pck_50': None, 'pck_5': None}
        self.metrics_agg = {k: None for k in self.which}
        
        # for diff sequence length, make a aggregation dict self.summaries
        # until 2, 4, 8, 10, 14, 25 frames
        # {'positional': {8: None, 12: None}, 'joint_angle': {8: None, 12: None}, 'euler': {8: None, 12: None}, 'pck_50': {8: None, 12: None}, 'pck_5': {8: None, 12: None}}
        self.summaries = {k: {t: None for t in target_lengths} for k in self.which}
        self.best_summaries = {k: {t: float('inf') for t in target_lengths} for k in self.which}

    def reset(self):
        '''
        Reset all metrics.
        '''
        self.metrics_agg = {k: None for k in self.which}
        self.n_samples = 0
        self._should_call_reset = False  # now it's again safe to compute new values
    
    def get_summary_feed_dict(self, final_metrics):
        """
        Compute the metrics for the target sequence lengths and return the feed dict that can be used in a call to
        `sess.run` to retrieve the Tensorboard summary ops.
        Args:
            final_metrics: Dictionary of metric values, expects them to be in shape (seq_length, ) except for PCK.

        Returns:
            The feed dictionary filled with values per summary.
        """
        feed_dict = dict()
        record_flag = False
        for m in self.summaries:
            for t in self.summaries[m]:
                name = "{}/until_{}".format(m, t)
                if m.startswith("pck"):
                    # does not make sense to sum up for pck
                    val = torch.mean(final_metrics[m][:t])
                else:
                    val = torch.sum(final_metrics[m][:t])
                feed_dict[name] = val
                if feed_dict[name] < self.best_summaries[m][t]:
                    self.best_summaries[m][t] = feed_dict[name]
                    record_flag = True

        return feed_dict, record_flag

    def compute_rotmat(self, predictions, targets, reduce_fn="mean"):
        '''
        Compute the chosen metrics. Predictions and targets are assumed to be in rotation matrix format.
        
        Args:
            - predictions: An tensor of shape (batch_size, seq_length, num_joints, 9)
            - targets: An tensor of the same shape as `predictions`
            - reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an tensor
            of shape (batch_size, seq_length). `reduce_fn` is only applied to metrics where it makes sense, i.e. not to PCK
            and euler angle differences.
        '''
        predictions = predictions.reshape(predictions.shape[0], predictions.shape[1], -1)
        targets = targets.reshape(targets.shape[0], targets.shape[1], -1)
        assert predictions.shape[-1] % 9 == 0, "predictions are not rotation matrices"
        assert targets.shape[-1] % 9 == 0, "targets are not rotation matrices"
        assert reduce_fn in ["mean", "sum"]
        assert not self._should_call_reset, "you should reset the state of this class after calling `finalize`"
        dof = 9
        n_joints = len(self.fk_engine.major_joints) if self.is_sparse else self.fk_engine.n_joints
        batch_size = predictions.shape[0]
        seq_length = predictions.shape[1]
        assert n_joints*dof == predictions.shape[-1], "unexpected number of joints"

        # first reshape everything to (-1, n_joints * 9)
        pred = predictions.reshape(-1, n_joints*dof).clone()
        targ = targets.reshape(-1, n_joints*dof).clone()

        # enforce valid rotations
        if self.force_valid_rot:
            pred_val = pred.reshape(-1, n_joints, 3, 3)
            pred = get_closest_rotmat(pred_val)
            pred = pred.reshape(-1, n_joints*dof)

        # check that the rotations are valid
        pred_are_valid = is_valid_rotmat(pred.reshape(-1, n_joints, 3, 3))
        assert pred_are_valid, 'predicted rotation matrices are not valid'
        targ_are_valid = is_valid_rotmat(targ.reshape(-1, n_joints, 3, 3))
        assert targ_are_valid, 'target rotation matrices are not valid'

        # add potentially missing joints
        if self.is_sparse:
            pred = sparse_to_full(pred, self.fk_engine.major_joints, self.fk_engine.n_joints, rep="rotmat")
            targ = sparse_to_full(targ, self.fk_engine.major_joints, self.fk_engine.n_joints, rep="rotmat")

        # make sure we don't consider the root orientation
        assert pred.shape[-1] == self.fk_engine.n_joints*dof
        assert targ.shape[-1] == self.fk_engine.n_joints*dof
        pred[:, 0:9] = torch.eye(3, 3).flatten()
        targ[:, 0:9] = torch.eye(3, 3).flatten()

        metrics = dict()

        if "positional" in self.which or "pck" in self.which:
            # need to compute positions - only do this once for efficiency
            pred_pos = self.fk_engine.from_rotmat(pred)  # (-1, full_n_joints, 3)
            targ_pos = self.fk_engine.from_rotmat(targ)  # (-1, full_n_joints, 3)
        else:
            pred_pos = targ_pos = None
        
        # default self.is_sparse = True
        select_joints = self.fk_engine.major_joints if self.is_sparse else list(range(self.fk_engine.n_joints))
        reduce_fn_np = torch.mean if reduce_fn == "mean" else torch.sum

        for metric in self.which:
            if metric.startswith("pck"):
                thresh = float(metric.split("_")[-1])
                v = pck(pred_pos[:, select_joints], targ_pos[:, select_joints], thresh=thresh)  # (-1, )
                metrics[metric] = v.reshape(batch_size, seq_length)
            elif metric == "positional":
                v = positional(pred_pos[:, select_joints], targ_pos[:, select_joints])  # (-1, n_joints)
                v = v.reshape(batch_size, seq_length, n_joints)
                metrics[metric] = reduce_fn_np(v, dim=-1)
            elif metric == "joint_angle":
                # compute the joint angle diff on the global rotations, not the local ones, which is a harder metric
                pred_global = local_rot_to_global(pred, self.fk_engine.parents, left_mult=self.fk_engine.left_mult,
                                                  rep="rotmat")  # (-1, full_n_joints, 3, 3)
                targ_global = local_rot_to_global(targ, self.fk_engine.parents, left_mult=self.fk_engine.left_mult,
                                                  rep="rotmat")  # (-1, full_n_joints, 3, 3)
                v = angle_diff(pred_global[:, select_joints], targ_global[:, select_joints])  # (-1, n_joints)
                v = v.reshape(batch_size, seq_length, n_joints)
                metrics[metric] = reduce_fn_np(v, dim=-1)
            elif metric == "euler":
                # compute the euler angle error on the local rotations, which is how previous work does it
                pred_local = pred.reshape(-1, self.fk_engine.n_joints, 3, 3)
                targ_local = targ.reshape(-1, self.fk_engine.n_joints, 3, 3)
                v = euler_diff(pred_local[:, select_joints], targ_local[:, select_joints])  # (-1, )
                metrics[metric] = v.reshape(batch_size, seq_length)
            else:
                raise ValueError("metric '{}' unknown".format(metric))

        return metrics

    def compute_quat(self, predictions, targets, reduce_fn="mean"):
        '''
        Compute the chosen metrics. Predictions and targets are assumed to be quaternions.
        
        Args:
            - predictions: An np array of shape (n, seq_length, n_joints*4)
            - targets: An np array of the same shape as `predictions`
            - reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an np array
            of shape (n, seq_length). `reduce_fn` is only applied to metrics where it makes sense, i.e. not to PCK
            and euler angle differences.
        '''
        assert predictions.shape[-1] % 4 == 0, "predictions are not quaternions"
        assert targets.shape[-1] % 4 == 0, "targets are not quaternions"
        assert reduce_fn in ["mean", "sum"]
        assert not self._should_call_reset, "you should reset the state of this class after calling `finalize`"
        dof = 4
        batch_size = predictions.shape[0]
        seq_length = predictions.shape[1]

        # for simplicity we just convert quaternions to rotation matrices
        pred_q = quaternion.from_float_array(np.reshape(predictions, [batch_size, seq_length, -1, dof]))
        targ_q = quaternion.from_float_array(np.reshape(targets, [batch_size, seq_length, -1, dof]))
        pred_rots = quaternion.as_rotation_matrix(pred_q)
        targ_rots = quaternion.as_rotation_matrix(targ_q)

        preds = np.reshape(pred_rots, [batch_size, seq_length, -1])
        targs = np.reshape(targ_rots, [batch_size, seq_length, -1])
        return self.compute_rotmat(preds, targs, reduce_fn)

    def compute_aa(self, predictions, targets, reduce_fn="mean"):
        '''
        Compute the chosen metrics. Predictions and targets are assumed to be in angle-axis format.
        
        Args:
            - predictions: An np array of shape (n, seq_length, n_joints*3)
            - targets: An np array of the same shape as `predictions`
            - reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an np array
            of shape (n, seq_length). `reduce_fn` is only applied to metrics where it makes sense, i.e. not to PCK
            and euler angle differences.
        '''
        assert predictions.shape[-1] % 3 == 0, "predictions are not quaternions"
        assert targets.shape[-1] % 3 == 0, "targets are not quaternions"
        assert reduce_fn in ["mean", "sum"]
        assert not self._should_call_reset, "you should reset the state of this class after calling `finalize`"
        dof = 3
        batch_size = predictions.shape[0]
        seq_length = predictions.shape[1]

        # for simplicity we just convert angle-axis to rotation matrices
        pred_aa = np.reshape(predictions, [batch_size, seq_length, -1, dof])
        targ_aa = np.reshape(targets, [batch_size, seq_length, -1, dof])
        pred_rots = aa2rotmat(pred_aa)
        targ_rots = aa2rotmat(targ_aa)
        preds = np.reshape(pred_rots, [batch_size, seq_length, -1])
        targs = np.reshape(targ_rots, [batch_size, seq_length, -1])
        return self.compute_rotmat(preds, targs, reduce_fn)

    def compute(self, predictions, targets, reduce_fn="mean"):
        '''
        Compute the chosen metrics. Predictions and targets can be in rotation matrix or quaternion format.
        
        Args:
            - predictions: An tensor of shape (batch_size, seq_length, num_joints, rep)
            - targets: An tensor of the same shape as `predictions`
            - reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an tensor
            of shape (batch_size, seq_length). `reduce_fn` is only applied to metrics where it makes sense, i.e. not to PCK
            and euler angle differences.
        '''
        if self.rep == "rotmat":
            return self.compute_rotmat(predictions, targets, reduce_fn)
        elif self.rep == "quat":
            return self.compute_quat(predictions, targets, reduce_fn)
        else:
            return self.compute_aa(predictions, targets, reduce_fn)

    def aggregate(self, new_metrics):
        '''
        Aggregate the metrics.
        
        Args:
            - new_metrics: Dictionary of new metric values to aggregate. Each entry is expected to be a tensor
                of shape (batch_size, seq_length). For PCK values there might be more than 2 dimensions.
        '''
        assert isinstance(new_metrics, dict)
        assert list(new_metrics.keys()) == list(self.metrics_agg.keys())

        # sum over the batch dimension
        for m in new_metrics:
            if self.metrics_agg[m] is None:
                self.metrics_agg[m] = torch.sum(new_metrics[m], dim=0)
            else:
                self.metrics_agg[m] += torch.sum(new_metrics[m], dim=0)

        # keep track of the total number of samples processed
        batch_size = new_metrics[list(new_metrics.keys())[0]].shape[0]
        self.n_samples += batch_size

    def compute_and_aggregate(self, predictions, targets, reduce_fn="mean"):
        '''
        Computes the metric values and aggregates them directly.
        
        Args:
            - predictions: An tensor of shape (n, seq_length, num_joints, rep)
            - targets: An tensor of the same shape as `predictions`
            - reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].
        '''
        new_metrics = self.compute(predictions, targets, reduce_fn)
        self.aggregate(new_metrics)

    def get_final_metrics(self):
        '''
        Finalize and return the metrics - this should only be called once all the data has been processed.
        
        Returns:
            A dictionary of the final aggregated metrics per time step.
        '''
        self._should_call_reset = True  # make sure to call `reset` before new values are computed
        assert self.n_samples > 0

        for m in self.metrics_agg:
            self.metrics_agg[m] = self.metrics_agg[m] / self.n_samples

        # return a copy of the metrics so that the class can be re-used again immediately
        return copy.deepcopy(self.metrics_agg)

    @classmethod
    def get_metrics_until(cls, metric_results, until, pck_thresholds=None,
                          prefix="", at_mode=False):
        '''
        Calculates the metrics at a given time-step.
        
        Args:
            - metric_results: Dictionary of metric values, expects them to be in
                shape (seq_length, ) except for PCK.
            - until: time-step to report metrics.
            - pck_thresholds: if not passed, then pck and auc are ignored.
            - prefix: a string identifier for metric keys such as 'test'.
            - at_mode: If true will report the numbers at the last frame rather
                than until the last frame.
        Returns:
            dict of metrics.
        '''
        metrics = dict()
        for m in sorted(metric_results):
            if not m.startswith("pck"):
                val = metric_results[m][until - 1] if at_mode else np.sum(
                        metric_results[m][:until])
                key = prefix + m
                metrics[key] = val

        if pck_thresholds is not None:
            pck_values = []
            for threshold in sorted(pck_thresholds):
                # Convert pck value in float to a str compatible name.
                t = threshold*100
                m_name = "pck_{}".format(t if t < 1 else (int(t)))
                val = metric_results[m_name][until - 1] if at_mode else np.mean(
                        metric_results[m_name][:until])
                metrics[prefix + m_name] = val
                pck_values.append(val)

            auc_val = cls.calculate_auc(pck_values, pck_thresholds, until)
            metrics[prefix + "AUC"] = auc_val
        return metrics
    
    @classmethod
    def get_summary_string(cls, metric_results, at_mode=False):
        '''
        Create a summary string (e.g. for printing to the console) from the
        given metrics for the entire sequence.
        
        Args:
            - metric_results: Dictionary of metric values, expects them to be in
                shape (seq_length, ) except for PCK.
            - at_mode: If true will report the numbers at the last frame rather
                than until the last frame.

        Returns:
            A summary string.
        '''
        seq_length = metric_results[list(metric_results.keys())[0]].shape[0]
        s = "metrics until {}:".format(seq_length)
        for m in sorted(metric_results):
            if m.startswith("pck"):
                continue
            val = metric_results[m][seq_length - 1] if at_mode else np.sum(
                    metric_results[m])
            s += "   {}: {:.3f}".format(m, val)

        # print pcks last
        pck_threshs = [0.05, 0.1, 0.15]
        for t in pck_threshs:
            t = t*100
            m_name = "pck_{}".format(t)
            val = metric_results[m_name][seq_length - 1] if at_mode else np.mean(metric_results[m_name])
            s += "   {}: {:.3f}".format(m_name, val)
        return s

    @classmethod
    def get_summary_string_all(cls, metric_results, target_lengths,
                               pck_thresholds, at_mode=False, report_pck=False):
        '''
        Create a summary string for given lengths. Note that this yields results
        reported in the paper.
        
        Args:
            - metric_results: Dictionary of metric values, expects them to be in
                shape (seq_length, ) except for PCK.
            - target_lengths: Metrics at these time-steps are reported.
            - pck_thresholds: PCK for this threshold values is reported.
            - at_mode: If true will report the numbers at the last frame rather
                than until the last frame.
            - report_pck: Whether to print all PCK values or not.

        Returns:
            A summary string.
        '''
        s = ""
        for i, seq_length in enumerate(sorted(target_lengths)):
            s += "Metrics until {:<2}:".format(seq_length)
            for m in sorted(metric_results):
                if m.startswith("pck"):
                    continue
                val = metric_results[m][seq_length - 1] if at_mode else np.sum(
                        metric_results[m][:seq_length])
                s += "   {}: {:.3f}".format(m, val)
        
            pck_values = []
            for threshold in sorted(pck_thresholds):
                # Convert pck value in float to a str compatible name.
                t = threshold*100
                m_name = "pck_{}".format(t if t < 1 else (int(t)))
                val = metric_results[m_name][seq_length - 1] if at_mode else np.mean(metric_results[m_name][:seq_length])
                if report_pck:
                    s += "   {}: {:.3f}".format(m_name, val)
                pck_values.append(val)

            auc = cls.calculate_auc(pck_values, pck_thresholds, seq_length)
            s += "   AUC: {:.3f}".format(auc)
            if i + 1 < len(target_lengths):
                s += "\n"
            
        return s
    
    @classmethod
    def calculate_auc(cls, pck_values, pck_thresholds, target_length):
        '''
        Calculate area under a curve (AUC) metric for PCK.
        
        If the sequence length is shorter, we ignore some of the high-tolerance PCK values in order to have less
        saturated AUC.
        
        Args:
            - pck_values (list): PCK values.
            - pck_thresholds (list): PCK threshold values.
            - target_length (int): determines for which time-step we calculate AUC.
        
        Returns:
        '''

        # Due to the saturation effect, we consider a limited number of PCK
        # thresholds in AUC calculation.
        if target_length < 6:
            n_pck = 6
        elif target_length < 12:
            n_pck = 7
        elif target_length < 18:
            n_pck = 8
        else:
            n_pck = len(pck_thresholds)
            
        norm_factor = np.diff(pck_thresholds[:n_pck]).sum()
        auc_values = []
        for i in range(n_pck - 1):
            auc = (pck_values[i] + pck_values[i + 1]) / 2 * (pck_thresholds[i + 1] - pck_thresholds[i])
            auc_values.append(auc)
        return np.array(auc_values).sum() / norm_factor
