import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import time
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.utils import weight_norm
from model.attn_module import *
from model.prompt_layers import *



class prompt_transformer(nn.Module):   #[358, 307, 883, 170]
    def __init__(self, data_config, training_config, DEVICE, d_k, d_v, n_heads, num_of_d,  region_cluster, max_n_num=883, hidden_dim=512, dropout=0.1, exp_factor=8, depth=1):
        super(prompt_transformer, self).__init__()
        self.data_config = data_config
        self.training_config = training_config
        self.node_keys = data_config['node_keys']
        self.depth = depth
        self.exp_factor = exp_factor
        self.dropout = dropout
        self.hidden_dim = training_config['d_model']
        self.tp = True
        self.sp = True
        self.his_num = training_config['points_per_hour']
        self.pred_num = training_config['num_for_predict']
        self.st_fusion = nn.Linear(self.hidden_dim, 1)
        self.linear1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear_spatial_init = nn.Linear(1, self.hidden_dim)
        self.w_s = torch.nn.Parameter(torch.Tensor(self.hidden_dim, 12*self.hidden_dim))
        self.w_t = torch.nn.Parameter(torch.Tensor(self.hidden_dim, 883*self.hidden_dim))
        self.linear_spatial = nn.Linear(self.his_num, self.hidden_dim)
        self.linear_temporal = torch.nn.Parameter(torch.Tensor(max_n_num * 2, self.hidden_dim))
        self.dropout_func = nn.Dropout(self.dropout)
        self.linear_weight_spatial = torch.nn.Parameter(torch.Tensor(self.hidden_dim, max_n_num * self.hidden_dim))
        self.linear_weight_temporal = nn.Linear(self.hidden_dim, self.his_num * self.hidden_dim)
        self.d_k = d_k 
        self.d_v = d_v 
        self.n_heads = n_heads 
        self.num_of_d = num_of_d
        self.region_cluster = region_cluster

        self.DEVICE = DEVICE
        self.reset_parameters()
        self.build()
    
    def build(self):

        self.predictor = nn.Linear(self.his_num, self.pred_num)
        #self.predictor = nn.Linear(self.hidden_dim, self.task_args['pred_num'])
        self.sp_learner = GATConv(self.hidden_dim * self.his_num, self.his_num, 3, False, dropout=0.1)
        


        self.encoder_spatial = MultiHeadAttention(self.DEVICE, self.hidden_dim * 2, self.d_k, self.d_v, self.n_heads, self.num_of_d)
        self.encoder_temporal = MultiHeadAttention(self.DEVICE, self.hidden_dim * 2, self.d_k, self.d_v, self.n_heads, self.num_of_d)
        self.pos_emb = Embedding(self.hidden_dim)

        self.prompt_layer = prompt_layer_weighted_node(self.hidden_dim, self.training_config)
        self.prompt_structure_layer = prompt_layer_feature_weighted_sum(self.hidden_dim)
        self.prompt_structure_layer_ = prompt_layer_feature_weighted_sum_(self.his_num)
        self.temperature = 0.2
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.sim = torch.nn.CosineSimilarity(-1)
        
        if self.tp and self.sp:
            self.alpha = nn.Parameter(torch.FloatTensor(self.hidden_dim))
            stdv = 1. / math.sqrt(self.alpha.shape[0])
            self.alpha.data.uniform_(-stdv, stdv)
        
        if self.tp == False and self.sp == False:
            print("sp and tp are all False.")
            self.meta_knowledge = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_s)
        torch.nn.init.xavier_uniform_(self.w_t)
        torch.nn.init.xavier_uniform_(self.linear_temporal)
        torch.nn.init.xavier_uniform_(self.linear_weight_spatial)


        
    def forward(self, x, stage='source'):

        region_index = self.region_cluster[x.shape[1]]
        region_index = torch.LongTensor(region_index).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)
        x = x.transpose(2,3)  #[32,358,1,12]
        cite_index = self.node_keys.index(x.shape[1])
        city_index = torch.LongTensor([cite_index]).repeat(x.shape[0]).to(x.device)


        embedding_s, embedding_t = x, x

        '''    
        change the shape of data.x into low dim
        sp:[b,node_num,seq_len,d]->[b,node_num,dim]
        tp:[b,node_num,seq_len,d]->[b,seq_len,dim]
        '''
        #embedding -> [btach_size, node_num, his_len, hidden_dim]
        batch_size, node_num, seq_len, hidden_dim = embedding_s.shape

        data_init = self.linear_spatial_init(embedding_s)
        #embedding_s => [batch_size, N, T*d]
        embedding_s = embedding_s.reshape(batch_size, node_num, seq_len*hidden_dim)
        data_spatial = self.linear_spatial(embedding_s)  #[b,node_num,dim]
        #embedding_t => [batch_size, T, N*d]
        embedding_t = embedding_t.transpose(1,2).reshape(batch_size, seq_len, node_num*hidden_dim)
        weight_t_ = self.linear_temporal[0:node_num*hidden_dim].unsqueeze(0)
        data_temporal = torch.matmul(embedding_t, weight_t_)  #[b,seq_len,dim]

        spatial, temporal = [], [] #[data_spatial.unsqueeze(-1)], [data_temporal.unsqueeze(-1)]
        fusion_tensor = []
        '''
        # ST encoder module
        # '''
        depth = 4
        for layer_num in range(depth):
            data_spatial, data_temporal = self.prompt_layer(data_spatial, data_temporal, city_index, region_index, layer_num)
            batch_size, seq_len, t_dim = data_temporal.shape
            batch_size, node_num, s_dim = data_spatial.shape
            if stage == 'source':
                data_spatial, data_temporal = self.submodule1(data_spatial, data_temporal, batch_size, seq_len, node_num)
            else:
                #with torch.no_grad():
                data_spatial, data_temporal = self.submodule1(data_spatial, data_temporal, batch_size, seq_len, node_num)
            spatial.append(data_spatial.unsqueeze(-1))
            temporal.append(data_temporal.unsqueeze(-1))
        data_spatial = torch.sum(torch.stack(spatial, dim=-1), dim=-1).squeeze()
        data_temporal = torch.sum(torch.stack(temporal,dim=-1), dim=-1).squeeze()

        if stage == 'source':
            sp_output, tp_output = self.submodule3(data_spatial, data_temporal, batch_size, node_num, seq_len)
        else:
            #with torch.no_grad():
            sp_output, tp_output = self.submodule3(data_spatial, data_temporal, batch_size, node_num, seq_len)

        sp = sp_output.reshape(sp_output.shape[0], sp_output.shape[1], -1)
        tp = tp_output.reshape(tp_output.shape[0], tp_output.shape[1], -1)
        pos_s = torch.zeros(sp.shape[1]).to(self.DEVICE).unsqueeze(0).unsqueeze(2)
        pos_t = torch.ones(sp.shape[1]).to(self.DEVICE).unsqueeze(0).unsqueeze(2)
        sp = self.ST2R(sp, pos_s, hidden_dim, x.device)
        tp = self.ST2R(tp, pos_t, hidden_dim, x.device)
        neg_index = torch.randperm(batch_size)
        shuf_s = sp[neg_index]
        shuf_t= tp[neg_index]
        #calculate the cosine similarity
        similarity_pos = self.sim(sp, tp)
        similarity_neg_s = self.sim(sp, shuf_s)
        similarity_neg_t = self.sim(tp, shuf_t)
        logits = torch.cat([similarity_pos, similarity_neg_s, similarity_neg_t], dim=0)
        logits /= self.temperature

        zero = torch.zeros(batch_size).to(self.DEVICE).long()
        one = torch.ones(2*batch_size).to(self.DEVICE).long()
        labels = torch.cat([zero, one], dim=0).long()
        loss = self.criterion(logits, labels)
        #loss = loss / (3 * batch_size)

        
        #[b,node_num,seq_len]        
        if self.tp and self.sp:
            output_ = torch.sigmoid(self.alpha) * sp_output + (1-torch.sigmoid(self.alpha)) * tp_output
            #gate fusion
            #z = torch.sigmoid(torch.add(sp_output, tp_output))
            # z = torch.sigmoid(output_)
            # output_ = torch.add(torch.mul(z, sp_output), torch.mul(1 - z, tp_output))
        elif self.tp:
            output_ = tp_output
        elif self.sp:
            output_ = sp_output
        


        
        #output_ = torch.cat([sp_output, tp_output], dim=-1)

        #output:[b,node_num,seq_len]->[b,node_num,dim]
        
        output = self.st_fusion(output_).squeeze()
        output_g = torch.reshape(output, (batch_size, node_num, self.his_num))



        output = F.relu(output)
        output = self.predictor(output)
        return output, loss

    def submodule1(self, data_spatial, data_temporal, batch_size, seq_len, node_num):
        mask = torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1).unsqueeze(0).repeat(batch_size,1,1).to(data_spatial.device)
        pos_s = torch.zeros(data_spatial.shape[1]).to(self.DEVICE).unsqueeze(0).unsqueeze(2)
        pos_t = torch.ones(data_temporal.shape[1]).to(self.DEVICE).unsqueeze(0).unsqueeze(2)
        data_spatial = self.ST2R(data_spatial, pos_s, self.hidden_dim, x.device)
        data_temporal = self.ST2R(data_temporal, pos_t, self.hidden_dim, x.device)
        data_spatial = self.encoder_spatial(data_spatial, data_spatial, data_spatial, None, None)
        data_temporal = self.encoder_temporal(data_temporal, data_temporal, data_temporal, mask, None)   #input_Q, input_K, input_V, attn_mask, res_att

        data_spatial = self.linear1(data_spatial).squeeze()
        data_temporal = self.linear2(data_temporal).squeeze()

        return data_spatial, data_temporal
    
    def submodule3(self, data_spatial, data_temporal, batch_size, node_num, seq_len):
        data_spatial = torch.matmul(data_spatial, self.w_s[:, 0:(seq_len*self.hidden_dim)])
        data_temporal = torch.matmul(data_temporal, self.w_t[:, 0:(node_num*self.hidden_dim)])
        sp_output = data_spatial.reshape(data_spatial.shape[0], node_num, seq_len, -1)
        #sp_output = sp_output.reshape(batch_size, node_num, seq_len)
        tp_output = data_temporal.reshape(data_temporal.shape[0], seq_len, node_num, -1).transpose(1, 2)

        return sp_output, tp_output
    
    def ST2R(self, x, positions, dim, device):

        # Ensure dimension is even for RoPE
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        
        # Generate position indices for the sequence
        # positions = torch.arange(seq_len, device=device).unsqueeze(1)  # (batch_size, seq_len, 1)
        
        # Compute frequency components
        freq_indices = torch.arange(0, dim // 2, device=device).unsqueeze(0).unsqueeze(0)  # (batch_sizeb, 1, dim//2)
        freq = 1.0 / (10000 ** (2 * freq_indices / dim))  # (batch_size, 1, dim//2)
        
        # Compute angle (theta) for sinusoidal components
        angle = positions * freq  # (batch_size, seq_len, dim//2)
        
        # Generate RoPE embeddings
        sin_enc = torch.sin(angle)  # (batch_size, seq_len, dim//2)
        cos_enc = torch.cos(angle)  # (batch_size, seq_len, dim//2)
        
        
        # Split the input tensor into real and imaginary parts
        x_real, x_imag = torch.chunk(x, 2, dim=-1)  # Each has shape (batch_size, seq_len, dim//2)
        
        # Apply the rotary transformation
        x_rotated_real = x_real * cos_enc - x_imag * sin_enc
        x_rotated_imag = x_real * sin_enc + x_imag * cos_enc
        
        # Combine back into one tensor
        x_rotated = torch.cat([x_rotated_real, x_rotated_imag], dim=-1)  # (batch_size, seq_len, dim)
        
        return x_rotated



