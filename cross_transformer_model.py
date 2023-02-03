from logging import raiseExceptions
import math
# from msilib.schema import Error
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
from utils import trunc_normal_

def create_look_ahead_mask(obs_length):
    '''
    create a look ahead mask given a certain window length for temporal usage.

    Args:
        - obs_length: obs length

    Return: 
        - mask: the mask (window_length, window_length)
    '''
    mask = 1 - torch.tril(torch.ones((obs_length, obs_length)))
    return mask  # (obs_length, obs_length)

class position_encoding():
    '''
    position encoding for temporal usage
    in a nn Module manner(without backpropagate)
    '''
    def __init__(self, obs_length, d_model):
        self.obs_length = obs_length
        self.d_model = d_model

    def get_angles(self, pos, i):
        '''
        calculate the angles givin postion and i for the positional encoding formula

        Args:
            - pos: pos in the formula
            - i: i in the formula

        Return: 
            -  : angle rad
        '''
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
        return pos * angle_rates

    def forward(self):
        '''
        calculate the positional encoding given the self.'window length'

        Return: 
            - positional encoding (1, obs_length, 1, d_model)
        '''
        angle_rads = self.get_angles(np.arange(self.obs_length)[:, np.newaxis], np.arange(self.d_model)[np.newaxis, :])

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, :, np.newaxis, :]

        return torch.from_numpy(pos_encoding).type(torch.float32)  # (1, seq_len, 1, d_model)

    def __call__(self):
        res = self.forward()
        return res

class split_head():
    '''
    split the embedding vector to different heads for the spatial/temporal attention
    '''
    def __init__(self, num_heads, d_model_in, d_model_out = 0, fix_low_rank = False):
        self.num_heads = num_heads
        self.d_model_in = d_model_in
        self.d_model_out = d_model_out
        self.fix_low_rank = fix_low_rank

        assert((self.d_model_out == 0)^self.fix_low_rank), "you need to set fix_low_rank = True to use d_model_out"

        if self.fix_low_rank:
            self.depth = self.d_model_out // self.num_heads
        else:
            self.depth = self.d_model_in // self.num_heads
        
        # Leave the dimensional scaling to make_qkv(include fixing low rank bottleneck), remove this Linear layer
    
    def forward(self, inputs: torch.Tensor):
        '''
        perform spatial/temporal split.

        Args:
            - inputs: the embedding vector (batch_size, num_joints/seq_len, d_model_in)
    
        Return:
            - head_outputs: (batch_size, num_heads, num_joints/seq_len, depth)
        '''
        if len(inputs.shape) == 3:
            head_output = torch.reshape(inputs, (inputs.shape[0], inputs.shape[1], self.num_heads, self.depth)) # (batch_size, seq_len/num_joints, num_heads, depth)
        elif len(inputs.shape) == 4:
            head_output = torch.reshape(inputs, (inputs.shape[0], inputs.shape[1], inputs.shape[2], self.num_heads, self.depth)) # (batch_size, seq_len, num_joints, num_heads, depth)
        else:
            raise ValueError('too many dims')
        return head_output.transpose(-2,-3) # (batch_size, num_heads, num_joints/seq_len, depth)

    def __call__(self, inputs):
        res = self.forward(inputs)
        return res

class scaled_dot_product_attention(nn.Module):
    '''
    The basic scaled dot product attention mechanism introduced in the Transformer
    '''
    def __init__(self):
        super(scaled_dot_product_attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask):
        '''
        Args:
            - q: the query vectors matrix (batch_size, num_head, attn_dim, d_model/num_heads or dq_fix)
            - k: the key vector matrix (batch_size, num_head, attn_dim, d_model/num_heads or dk_fix)
            - v: the value vector matrix (batch_size, num_head, attn_dim, d_model/num_heads or dv_fix)
            - mask: a mask for temporal use

        Return: 
            - wrapper_tuple: the updated encoding and the attention weights matrix
        '''
        # Q x K^T
        matmul_qk = torch.matmul(q, torch.transpose(k, -2, -1))  # (batch_size, num_heads, attn_dim, attn_dim)

        # scale matmul_qk
        dk = k.shape[-1]
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        # add the mask to the scaled tensor, make sure to consider cls.
        if mask is not None:
            if len(scaled_attention_logits.shape) == 4:
                scaled_attention_logits[:, :, 1:, 1:] = scaled_attention_logits[:, :, 1:, 1:] + (mask * -1e9)
            elif len(scaled_attention_logits.shape) == 5:
                scaled_attention_logits[:, :, :, 1:, 1:] = scaled_attention_logits[:, :, :, 1:, 1:] + (mask * -1e9)
            else:
                raise ValueError('wrong dims')

        # normalized on the last axis so that the scores add up to 1.
        attention_weights: torch.Tensor = self.softmax(scaled_attention_logits)  # (..., num_heads, attn_dim, attn_dim)
        # attention_weights = self.dropout(attention_weights)
        # attention_weight * V
        output = torch.matmul(attention_weights, v)  # (..., num_heads, attn_dim, depth)

        wrapper_list = [output, attention_weights]
        wrapper_tuple = tuple(wrapper_list)
        return wrapper_tuple

class joint_wise_ffn(nn.Module):
    '''
    The feed forward network
    '''
    def __init__(self, num_joints, d_model, intermedia_dim):
        super(joint_wise_ffn, self).__init__()
        self.NUM_JOINTS = num_joints
        self.d_model = d_model
        self.intermedia_dim = intermedia_dim
        self.embedding_lays = nn.ModuleList()
        for i in range(0, self.NUM_JOINTS):
            embedding_layer1 = nn.Linear(self.d_model, self.intermedia_dim)
            embedding_layer2 = nn.Linear(self.intermedia_dim, self.d_model)
            embedding_layer_tuple = nn.ModuleList()
            embedding_layer_tuple.append(embedding_layer1)
            embedding_layer_tuple.append(embedding_layer2)
            self.embedding_lays.append(embedding_layer_tuple)

    def forward(self, inputs:torch.Tensor):
        '''
        Perform as a FFN in standard Transformer

        Args: 
            - inputs: inputs (batch_size, seq_len, num_joints, d_model)
    
        return: 
            - outputs (batch_size, seq_len, num_joints, d_model)
        '''
        inputs = inputs.permute(2, 0, 1, 3)  # (num_joints, batch_size, seq_len, d_model)
        outputs = []
        # different joints have different embedding matrices
        for idx in range(self.NUM_JOINTS):
            joint_outputs = self.embedding_lays[idx][0](inputs[idx])
            joint_outputs = Func.relu(joint_outputs)
            joint_outputs = self.embedding_lays[idx][1](joint_outputs)
            outputs += [joint_outputs]
        outputs: torch.Tensor = torch.cat(outputs, axis=-1)  # (batch_size, seq_len, num_joints * d_model)
        outputs = torch.reshape(outputs, (outputs.shape[0], outputs.shape[1], self.NUM_JOINTS, self.d_model)) # (batch_size, seq_len, num_joints, d_model)
        return outputs

class temporal_attention(nn.Module):
    '''
    the temporal attention block with multi-head, under the situation of self-attention
    '''
    def __init__(self, num_heads, d_model, num_joints, dq_out, dk_out, dv_out, shared_templ_qkv, seq_len):
        super(temporal_attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.NUM_JOINTS = num_joints
        self.dq_out = dq_out
        self.dk_out = dk_out
        self.dv_out = dv_out
        self.shared_templ_qkv = shared_templ_qkv
        self.seq_len = seq_len
        self.pos_encoding = position_encoding(self.seq_len, self.d_model)

        if self.shared_templ_qkv:
            self.linear_q = nn.Linear(self.d_model, self.dq_out if self.dq_out else self.d_model)
            self.linear_k = nn.Linear(self.d_model, self.dk_out if self.dk_out else self.d_model)
            self.linear_v = nn.Linear(self.d_model, self.dv_out if self.dv_out else self.d_model)
        else:
            self.q_embedding_lays = nn.ModuleList()
            self.k_embedding_lays = nn.ModuleList()
            self.v_embedding_lays = nn.ModuleList()
            for i in range(0, self.NUM_JOINTS):
                q_embedding_layer = nn.Linear(self.d_model, self.dq_out if self.dq_out else self.d_model)
                self.q_embedding_lays.append(q_embedding_layer)
                k_embedding_layer = nn.Linear(self.d_model, self.dk_out if self.dk_out else self.d_model)
                self.k_embedding_lays.append(k_embedding_layer)
                v_embedding_layer = nn.Linear(self.d_model, self.dv_out if self.dv_out else self.d_model)
                self.v_embedding_lays.append(v_embedding_layer)

        # add temporal with para dq_out & dk_out indicate higher dim to avoid low rank bottleneck
        if self.dq_out and self.dk_out and self.dv_out:
            self.temporal_split_heads_q = split_head(self.num_heads, self.d_model, self.dq_out, True)
            self.temporal_split_heads_k = split_head(self.num_heads, self.d_model, self.dk_out, True)
            self.temporal_split_heads_v = split_head(self.num_heads, self.d_model, self.dv_out, True)
            self.concat_linear = nn.Linear(self.dv_out, self.d_model)
        else:
            self.temporal_split_heads = split_head(self.num_heads, self.d_model)
            self.concat_linear = nn.Linear(self.d_model, self.d_model)
        
        self.scaled_dot_product_attention = scaled_dot_product_attention()

    def forward(self, inputs):
        '''
        temporal branch attention

        Args:
            - x: the inputs embedding (batch_size, seq_len, num_joints, d_model)
            - mask: temporal mask (usually the look ahead mask)
        
        Return: 
            - the output (batch_size, seq_len, num_joints, d_model)
        '''
        x, mask= inputs

        # add position encoding, NOTE remove the '.forward()' after 'self.pos_encoding'
        inp_seq_len = x.shape[1]
        x += self.pos_encoding.forward()[:, :inp_seq_len].cuda()

        # preparation
        attn_weights = []
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.permute(2, 0, 1, 3)  # (num_joints, batch_size, seq_len, d_model)
        # remove the loop
        # make q, k and v
        if self.shared_templ_qkv:
            q_all = self.linear_q(x)  # (num_joints, batch_size, seq_len, dq_out)
            k_all = self.linear_k(x)  # (num_joints, batch_size, seq_len, dk_out)
            v_all = self.linear_v(x)  # (num_joints, batch_size, seq_len, dv_out)
        else:
            q_joints = []
            k_joints = []
            v_joints = []
            # different joints may have different embedding matrices
            for joint_idx in range(self.NUM_JOINTS):
                # get the representation vector of the single joint
                joint_rep = x[joint_idx]  # (batch_size, seq_len, d_model)
                k = torch.unsqueeze(self.k_embedding_lays[joint_idx](joint_rep), axis=0)  # (1, batch_size, seq_len, dk_out)
                v = torch.unsqueeze(self.v_embedding_lays[joint_idx](joint_rep), axis=0)  # (1, batch_size, seq_len, dv_out)
                q = torch.unsqueeze(self.q_embedding_lays[joint_idx](joint_rep), axis=0)  # (1, batch_size, seq_len, dq_out)
                q_joints += [q]
                k_joints += [k]
                v_joints += [v]
            q_all: torch.Tensor = torch.cat(q_joints, axis=0)  # (num_joints, batch_size, seq_len, d_model)
            k_all: torch.Tensor = torch.cat(k_joints, axis=0)  # (num_joints, batch_size, seq_len, d_model)
            v_all: torch.Tensor = torch.cat(v_joints, axis=0)  # (num_joints, batch_size, seq_len, d_model)

        # split heads
        if self.dq_out and self.dk_out and self.dv_out:
            q_heads = self.temporal_split_heads_q(q_all.permute(1, 0, 2, 3))
            k_heads = self.temporal_split_heads_k(k_all.permute(1, 0, 2, 3))
            v_heads = self.temporal_split_heads_v(v_all.permute(1, 0, 2, 3))
        else:
            q_heads = self.temporal_split_heads(q_all.permute(1, 0, 2, 3))
            k_heads = self.temporal_split_heads(k_all.permute(1, 0, 2, 3))
            v_heads = self.temporal_split_heads(v_all.permute(1, 0, 2, 3))
        
        # calculate the updated encoding by scaled dot product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
        # (batch_size, num_joints, num_heads, seq_len, depth)
        scaled_attention = scaled_attention.permute(0, 3, 1, 2, 4)
        # (batch_size, seq_len, num_joints, num_heads, depth)

        # concatenate the outputs from different heads
        concat_attention = torch.reshape(scaled_attention, (batch_size, seq_len, self.NUM_JOINTS, -1))
        # (batch_size, seq_len, num_joints, d_model or depth * num_heads)

        # go through a fully connected layer, and unsqueeze the No.2 dim
        # in pst, each concat linear layer is different for different joint
        output = self.concat_linear(concat_attention)

        # last frame's attention weight shows how previous frames affect the last frame
        # last_attention_weights = attention_weights[:, :, -1, :]  # (batch_size, num_heads, seq_len)
        # attn_weights += [last_attention_weights]

        wrapper_list = [output, mask] # for nn.sequential use, dont return attention weight
        wrapper_tuple = tuple(wrapper_list)
        return wrapper_tuple
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=np.sqrt(1/math.sqrt(m.weight.shape[0] * m.weight.shape[1])))

class spatial_attention(nn.Module):
    '''
    the spatial attention block with multi-head, under the situation of self-attention
    '''
    def __init__(self, num_heads, d_model, num_joints, dq_out, dk_out, dv_out, shared_templ_qkv):
        super(spatial_attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.NUM_JOINTS = num_joints
        self.dq_out = dq_out
        self.dk_out = dk_out
        self.dv_out = dv_out
        self.shared_templ_qkv = shared_templ_qkv
        # in pst, for q, diff joint has diff embedding metric, k&v use one single linear layer respectively
        if self.shared_templ_qkv:
            self.linear_q = nn.Linear(self.d_model, self.dq_out if self.dq_out else self.d_model)
            self.linear_k = nn.Linear(self.d_model, self.dk_out if self.dk_out else self.d_model)
            self.linear_v = nn.Linear(self.d_model, self.dv_out if self.dv_out else self.d_model)
        else:
            self.q_embedding_lays = nn.ModuleList()
            self.k_embedding_lays = nn.ModuleList()
            self.v_embedding_lays = nn.ModuleList()
            for i in range(0, self.NUM_JOINTS):
                q_embedding_layer = nn.Linear(self.d_model, self.dq_out if self.dq_out else self.d_model)
                self.q_embedding_lays.append(q_embedding_layer)
                k_embedding_layer = nn.Linear(self.d_model, self.dk_out if self.dk_out else self.d_model)
                self.k_embedding_lays.append(k_embedding_layer)
                v_embedding_layer = nn.Linear(self.d_model, self.dv_out if self.dv_out else self.d_model)
                self.v_embedding_lays.append(v_embedding_layer)
        if self.dq_out and self.dk_out and self.dv_out:
            self.spatial_split_heads_q = split_head(self.num_heads, self.d_model, self.dq_out, True)
            self.spatial_split_heads_k = split_head(self.num_heads, self.d_model, self.dk_out, True)
            self.spatial_split_heads_v = split_head(self.num_heads, self.d_model, self.dv_out, True)
            self.concat_linear = nn.Linear(self.dv_out, self.d_model)
        else:
            self.spatial_split_heads = split_head(self.num_heads, self.d_model)
            self.concat_linear = nn.Linear(self.d_model, self.d_model)
        self.scaled_dot_product_attention = scaled_dot_product_attention()

    def forward(self, inputs):
        '''
        spatial branch attention

        Args:
            - x: the input (batch_size, seq_len, num_joints, d_model)
            - mask: spatial mask (usually None)

        Return: 
            - the output (batch_size, seq_len, num_joints, d_model)
        '''
        x, _ = inputs

        # embed each vector to key, value and query vectors
        if self.shared_templ_qkv:
            q_all = self.linear_q(x)  # (batch_size, seq_len, num_joints, d_model)
            k_all = self.linear_k(x)  # (batch_size, seq_len, num_joints, d_model)
            v_all = self.linear_v(x)  # (batch_size, seq_len, num_joints, d_model)
        else:
            x = x.permute(2, 0, 1, 3) # (num_joints, batch_size, seq_len, d_model)
            q_joints = []
            k_joints = []
            v_joints = []
            for joint_idx in range(self.NUM_JOINTS):
                q_joint = torch.unsqueeze(self.q_embedding_lays[joint_idx](x[joint_idx]), axis=2)  # (batch_size, seq_len, 1, d_model)
                q_joints += [q_joint]
                k_joint = torch.unsqueeze(self.k_embedding_lays[joint_idx](x[joint_idx]), axis=2)  # (batch_size, seq_len, 1, d_model)
                k_joints += [k_joint]
                v_joint = torch.unsqueeze(self.v_embedding_lays[joint_idx](x[joint_idx]), axis=2)  # (batch_size, seq_len, 1, d_model)
                v_joints += [v_joint]
            q_all: torch.Tensor = torch.cat(q_joints, axis=2)  # (batch_size, seq_len, num_joints, d_model)
            k_all: torch.Tensor = torch.cat(k_joints, axis=2)  # (batch_size, seq_len, num_joints, d_model)
            v_all: torch.Tensor = torch.cat(v_joints, axis=2)  # (batch_size, seq_len, num_joints, d_model)
        
        # preparation work
        batch_size = q_all.shape[0]
        seq_len = q_all.shape[1]
        last_frame_weight = None
        # remove the loop
        if self.dq_out and self.dk_out and self.dv_out:
            # split it to several attention heads
            q = self.spatial_split_heads_q(q_all)
            # (batch_size, seq_len, num_heads, num_joints, depth)
            k = self.spatial_split_heads_k(k_all)
            # (batch_size, seq_len, num_heads, num_joints, depth)
            v = self.spatial_split_heads_v(v_all)
            # (batch_size, seq_len, num_heads, num_joints, depth)
        else:
            # split it to several attention heads
            q = self.spatial_split_heads(q_all)
            # (batch_size, seq_len, num_heads, num_joints, depth)
            k = self.spatial_split_heads(k_all)
            # (batch_size, seq_len, num_heads, num_joints, depth)
            v = self.spatial_split_heads(v_all)
            # (batch_size, seq_len, num_heads, num_joints, depth)

        # calculate the updated encoding by scaled dot product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, None)
        scaled_attention = scaled_attention.permute(0, 1, 3, 2, 4)
        # (batch_size, seq_len, num_joints, num_heads, depth)

        # concatenate the outputs from different heads
        concat_attention = torch.reshape(scaled_attention, (batch_size, seq_len, self.NUM_JOINTS, -1))
        # (batch_size, seq_len, num_joints, d_model or dv_out)

        # go through a fully connected layer
        output = self.concat_linear(concat_attention)
        # (batch_size, seq_len, num_joints, d_model)
        last_frame_weight = attention_weights

        # wrapper_list = [outputs, last_frame_weight]
        wrapper_list = [output, None] # for nn.sequential use, dont return attention weight
        wrapper_tuple = tuple(wrapper_list)
        return wrapper_tuple

class cross_attention(nn.Module):
    def __init__(self, num_heads, d_model, dq_out, dk_out, dv_out, att_dim):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dq_out = dq_out
        self.dk_out = dk_out
        self.dv_out = dv_out
        # from d_model to depth x num_heads, fix_low_rank 
        self.wq = nn.Linear(self.d_model, self.dq_out if self.dq_out else self.d_model)
        self.wk = nn.Linear(self.d_model, self.dk_out if self.dk_out else self.d_model)
        self.wv = nn.Linear(self.d_model, self.dv_out if self.dv_out else self.d_model)

        if self.dq_out and self.dk_out and self.dv_out:
            self.split_heads_q = split_head(self.num_heads, self.d_model, self.dq_out, True)
            self.split_heads_k = split_head(self.num_heads, self.d_model, self.dk_out, True)
            self.split_heads_v = split_head(self.num_heads, self.d_model, self.dv_out, True)
            self.concat_linear = nn.Linear(self.dv_out, self.d_model)
        else:
            self.split_heads = split_head(self.num_heads, self.d_model)
            self.concat_linear = nn.Linear(self.d_model, self.d_model)
        self.scaled_dot_product_attention = scaled_dot_product_attention()
        self.att_dim = att_dim # 1 for Temporal, 2 for Spatial

    def forward(self, x):
        '''
        Cross attention, gather info from other branch.
        
        Args: 
            - x : [cls_t/s ; patch_s/t]
            - q : in crossattention refers to linear(cls_t/s)
            - k, v : in crossattention refers to linear([cls_t/s ; patch_s/t])

        Return: 
            - output : updated cls_t/s
        '''
        B, S, N, D = x.shape # B for batch, S for seq_len, N for number_joints, D for d_model
        if self.dq_out and self.dk_out and self.dv_out:
            if self.att_dim == 1:
                q = self.split_heads_q(self.wq(x[:, 0:1, ...]))  # B1ND -> B1H(C/H) -> BH1(C/H)
            elif self.att_dim == 2:
                q = self.split_heads_q(self.wq(x[:, :, 0:1, ...]))
            else:
                raise ValueError('cant perform attention on dimension:{}'.format(self.att_dim))
            k = self.split_heads_k(self.wk(x))
            v = self.split_heads_v(self.wv(x))
        else :
            if self.att_dim == 1:
                q = self.split_heads(self.wq(x[:, 0:1, ...]))  # BS1D -> B1H(C/H) -> BH1(C/H)
            elif self.att_dim == 2:
                q = self.split_heads(self.wq(x[:, :, 0:1, ...]))
            else:
                raise ValueError('cant perform attention on dimension:{}'.format(self.att_dim))
            k = self.split_heads(self.wk(x))
            v = self.split_heads(self.wv(x))

        if self.att_dim == 1:
            q = q.permute(0, 2, 3, 1, 4)
            k = k.permute(0, 2, 3, 1, 4)
            v = v.permute(0, 2, 3, 1, 4)
        elif self.att_dim == 2:
            q = q.permute(0, 2, 1, 3, 4)
            k = k.permute(0, 2, 1, 3, 4)
            v = v.permute(0, 2, 1, 3, 4)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, None)
        scaled_attention = scaled_attention.permute(0, 3, 2, 1, 4) if self.att_dim == 1 else scaled_attention.permute(0, 2, 3, 1, 4)

        # concatenate the outputs from different heads
        concat_attention = torch.reshape(scaled_attention, (B, 1, N, -1)) if self.att_dim == 1 else torch.reshape(scaled_attention, (B, S, 1, -1))
        # (batch_size, num_joints, d_model or depth * num_heads)

        # go through a fully connected layer
        output = self.concat_linear(concat_attention)
        # linear computational cost
        return output

class st_parallel_attention(nn.Module):
    '''
    The block with spatial and temporal and cross blocks in parallel
    '''
    def __init__(self, num_layers, temporal_num_heads, spatial_num_heads, d_model, num_joints, seq_len, dq_out, dk_out, dv_out, shared_templ_qkv, dropout_rate):
        super(st_parallel_attention, self).__init__()
        self.num_layers = num_layers
        self.temporal_num_heads = temporal_num_heads
        self.spatial_num_heads = spatial_num_heads
        self.d_model = d_model
        self.NUM_JOINTS = num_joints
        self.seq_len = seq_len
        self.dq_out = dq_out
        self.dk_out = dk_out
        self.dv_out = dv_out
        self.shared_templ_kv = shared_templ_qkv
        self.rate = dropout_rate
        N, L, M = self.num_layers
        temp_t = []
        temp_s = []
        for i in range(N):
            temp_t.append(temporal_attention(self.temporal_num_heads, self.d_model, self.NUM_JOINTS, self.dq_out, self.dk_out, self.dv_out, self.shared_templ_kv, self.seq_len + 1))
            temp_s.append(spatial_attention(self.spatial_num_heads, self.d_model, self.NUM_JOINTS + 1, self.dq_out, self.dk_out, self.dv_out, self.shared_templ_kv))
        self.temporal_attention_stageI = nn.Sequential(*temp_t)
        self.spatial_attention_stageI = nn.Sequential(*temp_s)
        temp_t = []
        temp_s = []
        for i in range(L):
            temp_t.append(cross_attention(self.temporal_num_heads, self.d_model, self.dq_out, self.dk_out, self.dv_out, 1))
            temp_s.append(cross_attention(self.temporal_num_heads, self.d_model, self.dq_out, self.dk_out, self.dv_out, 2))
        self.temporal_cross = nn.Sequential(*temp_t)
        self.spatial_cross = nn.Sequential(*temp_s)
        temp_t = []
        temp_s = []
        for i in range(M):
            temp_t.append(temporal_attention(self.temporal_num_heads, self.d_model, self.NUM_JOINTS, self.dq_out, self.dk_out, self.dv_out, self.shared_templ_kv, self.seq_len + 1))
            temp_s.append(spatial_attention(self.spatial_num_heads, self.d_model, self.NUM_JOINTS + 1, self.dq_out, self.dk_out, self.dv_out, self.shared_templ_kv))
        self.temporal_attention_stageII = nn.Sequential(*temp_t)
        self.spatial_attention_stageII = nn.Sequential(*temp_s)
        # add&layernorm, before and after self_attention
        self.ln_temporal_stageI = nn.LayerNorm([self.seq_len + 1, self.NUM_JOINTS, self.d_model], eps=1e-05, elementwise_affine=True)
        self.ln_spatial_stageI = nn.LayerNorm([self.seq_len, self.NUM_JOINTS + 1, self.d_model], eps=1e-05, elementwise_affine=True)
        self.ln_temporal_cross = nn.LayerNorm([1, self.NUM_JOINTS, self.d_model], eps=1e-05, elementwise_affine=True)
        self.ln_spatial_cross = nn.LayerNorm([self.seq_len, 1, self.d_model], eps=1e-05, elementwise_affine=True)
        self.ln_temporal_stageII = nn.LayerNorm([self.seq_len + 1, self.NUM_JOINTS, self.d_model], eps=1e-05, elementwise_affine=True)
        self.ln_spatial_stageII = nn.LayerNorm([self.seq_len, self.NUM_JOINTS + 1, self.d_model], eps=1e-05, elementwise_affine=True)

        # add 6 more joint_wise_ffn if nesessary
    def forward(self, x, look_ahead_mask):
        '''
        Args:
            - x: tuple() of [cls_t;patch_t](batch_size, seq_len + 1, num_joints, d_model) and [cls_s;patch_s](batch_size, seq_len, num_joints + 1, d_model)
            - look_ahead_mask: the look ahead mask
            
        Return: 
            - outputs : (batch_size, seq_len + 1, num_joints, d_model) & (batch_size, seq_len, num_joints + 1, d_model) and the attention blocks
        '''
        # stage I
        ## temporal attention 
        attn_1t, _ = self.temporal_attention_stageI((x[0], look_ahead_mask))
        # 1.dropout 2.layernorm
        attn_1t = self.ln_temporal_stageI(attn_1t + x[0])
        ## spatial attention 
        attn_1s, _ = self.spatial_attention_stageI((x[1], None))
        # 1.dropout 2.layernorm
        attn_1s = self.ln_spatial_stageI(attn_1s + x[1])

        # stage II
        ## temporal cross
        cross_t = self.temporal_cross(torch.cat((attn_1t[:, :1], attn_1s[:, :, 1:]), dim=1))
        attn_1t[:, :1] = self.ln_temporal_cross(cross_t + attn_1t[:, :1])
        ## spatial cross
        cross_s = self.spatial_cross(torch.cat((attn_1s[:, :, :1], attn_1t[:, 1:]), dim=2))
        attn_1s[:, :, :1] = self.ln_spatial_cross(cross_s + attn_1s[:, :, :1])

        # stage III
        ## temporal attention 
        attn_2t, _ = self.temporal_attention_stageII((attn_1t, look_ahead_mask))
        # 1.dropout 2.add&layernorm
        temporal_out = self.ln_temporal_stageII(attn_2t + attn_1t)
        ## spatial attention 
        attn_2s, _ = self.spatial_attention_stageII((attn_1s, None))
        # 1.dropout 2.add&layernorm
        spatial_out = self.ln_spatial_stageII(attn_2s + attn_1s)

        wrapper_list = [temporal_out, spatial_out] # , attn_weights_block2t, attn_weights_block2s
        wrapper_tuple = tuple(wrapper_list) 
        return wrapper_tuple

class TransformerModel(nn.Module):
    '''
    The whole transformer-based network
    '''
    def __init__(self, temporal_num_heads, spatial_num_heads, d_model, num_joints, rep_dim, seq_len, dq_out, dk_out, dv_out, shared_templ_qkv, dropout_rate, num_layers, residual_velocity = True):
        super(TransformerModel, self).__init__()
        self.temporal_num_heads = temporal_num_heads
        self.spatial_num_heads = spatial_num_heads
        self.d_model = d_model
        self.NUM_JOINTS = num_joints
        self.rep_dim = rep_dim
        self.seq_len = seq_len
        self.dq_out = dq_out
        self.dk_out = dk_out
        self.dv_out = dv_out
        self.shared_templ_kv = shared_templ_qkv
        self.rate = dropout_rate
        self.num_layers = num_layers
        self.residual_velocity = residual_velocity

        # make cls_t and cls_s
        self.cls_t = nn.Parameter(torch.zeros(1, self.NUM_JOINTS, self.d_model))
        trunc_normal_(self.cls_t, std=.02)
        self.cls_s = nn.Parameter(torch.zeros(self.seq_len, 1, self.d_model))
        trunc_normal_(self.cls_s, std=.02)

        # embedding
        self.embedding_layers = nn.ModuleList()
        for i in range(0, self.NUM_JOINTS):
            embedding_layer = nn.Linear(self.rep_dim, self.d_model)
            self.embedding_layers.append(embedding_layer)
        
        # dropout
        self.dropout = nn.Dropout(p = self.rate)

        # num * st_parallel_attention
        self.transformer_layers = nn.ModuleList()
        for i in range(len(self.num_layers)):
            st_parallel_transformer_layer: nn.Module = st_parallel_attention(self.num_layers[i], self.temporal_num_heads, self.spatial_num_heads, self.d_model, self.NUM_JOINTS, self.seq_len, self.dq_out, self.dk_out, self.dv_out, self.shared_templ_kv, self.rate)
            self.transformer_layers.append(st_parallel_transformer_layer)
        
        # FFN
        self.ffn = joint_wise_ffn(self.NUM_JOINTS, self.d_model, self.d_model)
        self.ln_ffn = torch.nn.LayerNorm([self.seq_len, self.NUM_JOINTS, self.d_model], eps=1e-05, elementwise_affine=True)
        # reverse embedding
        self.reverse_embedding_layers = nn.ModuleList()
        for i in range(0, self.NUM_JOINTS):
            reverse_embedding_layer = nn.Linear(self.d_model, self.rep_dim)
            self.reverse_embedding_layers.append(reverse_embedding_layer)
        
        # init weights
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs:torch.Tensor, look_ahead_mask):
        '''
        Args:
            - inputs: inputs (batch_size, seq_len, num_joints, joint_size)
            - look_ahead_mask: the look ahead mask for temporal
    
        Return: 
            - outputs (batch_size, seq_len, num_joints, joint_size)
        '''
        # encode each 3D pos to high dim feature space (d_model), different joints have different encoding matrices
        inputs = inputs.permute(2, 0, 1, 3)  # (num_joints, batch_size, seq_len, joint_size)
        embed = []
        for joint_idx in range(self.NUM_JOINTS):
            joint_rep = self.embedding_layers[joint_idx](inputs[joint_idx])  # (batch_size, seq_len, d_model)
            embed += [joint_rep]
        x = torch.stack(embed, dim=2)
        
        # dropout
        x = self.dropout(x)

        # batch cls_t/s, make xt/s
        cls_t = self.cls_t.expand(x.shape[0], -1, -1, -1)
        cls_s = self.cls_s.expand(x.shape[0], -1, -1, -1)
        xt = torch.cat((cls_t, x), dim=1)
        xs = torch.cat((cls_s, x), dim=2)

        # preparation work
        attention_weights_temporal = []
        attention_weights_spatial = []
        attention_weights = {}
       
        # go through several STPAs
        for idx in range(len(self.num_layers)):
            xt, xs = self.transformer_layers[idx]([xt, xs], look_ahead_mask)
            # attention_weights_temporal += [block1]  # (batch_size, num_joints, num_heads, seq_len)
            # attention_weights_spatial += [block2]  # (batch_size, num_heads, num_joints, num_joints)

        # attention_weights['temporal'] = torch.stack(attention_weights_temporal, axis=1)  # (batch_size, num_layers, num_joints, num_heads, seq_len)
        # attention_weights['spatial'] = torch.stack(attention_weights_spatial, axis=1)  # (batch_size, num_layers, num_heads, num_joints, num_joints)

        # residual and ffn
        out = xt[:, 1:] + xs[:, :, 1:]
        ffn_out = self.ffn(out)
        ln_out = self.ln_ffn(out + ffn_out)
        
        # decode each feature to the 3D pos space(project d_model back into joint_rep by joint)
        ln_out = ln_out.permute(2, 0, 1, 3)# (num_joints, batch_size, seq_len, d_model)
        output = []
        for joint_idx in range(self.NUM_JOINTS):
            joint_output = self.reverse_embedding_layers[joint_idx](ln_out[joint_idx])
            output += [joint_output]

        final_output = torch.cat(output, axis=-1)
        final_output = final_output.reshape(final_output.shape[0], final_output.shape[1], self.NUM_JOINTS, self.rep_dim)
        
        # global residual
        if self.residual_velocity:
            final_output += inputs.permute(1, 2, 0, 3)
        
        wrapper_list = [final_output, attention_weights]
        wrapper_tuple = tuple(wrapper_list)
        return wrapper_tuple

def get_model(args):
    if args.dataset_mode == 'amass':
        return TransformerModel(temporal_num_heads=args.num_heads_temporal,
                                spatial_num_heads=args.num_heads_spatial,
                                d_model=args.d_model,
                                num_joints=22,
                                rep_dim=3,
                                seq_len=args.obs_seq_len,
                                dq_out=args.dq,
                                dk_out=args.dk,
                                dv_out=args.dv,
                                shared_templ_qkv=args.shared_qkv,
                                dropout_rate=args.drop_rate,
                                num_layers=args.num_layers,
                                residual_velocity=args.residual_velocity).cuda()
    else :
        return TransformerModel(temporal_num_heads=args.num_heads_temporal,
                                spatial_num_heads=args.num_heads_spatial,
                                d_model=args.d_model,
                                num_joints=19,
                                rep_dim=3,
                                seq_len=args.obs_seq_len,
                                dq_out=args.dq,
                                dk_out=args.dk,
                                dv_out=args.dv,
                                shared_templ_qkv=args.shared_qkv,
                                dropout_rate=args.drop_rate,
                                num_layers=args.num_layers,
                                residual_velocity=args.residual_velocity).cuda()