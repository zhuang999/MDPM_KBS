import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def clones(module, num_sub_layer):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_sub_layer)])

class SubLayerConnect(nn.Module):
    def __init__(self, features):
        super(SubLayerConnect, self).__init__()
        

    def forward(self, x, sublayer):
        # (*, d)
        self.norm = nn.LayerNorm(x.shape[-1]).cuda()
        return x + sublayer(self.norm(x))

class FFN(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(FFN, self).__init__()
        #self.w_1 = torch.nn.Parameter(torch.Tensor(627*features, exp_factor * features))
        #nn.Linear(627*features, exp_factor * features)
        self.w_1 = nn.Linear(features, exp_factor * features)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(exp_factor * features, features)
        #self.w_2 = torch.nn.Parameter(torch.Tensor(exp_factor * features, 627*features))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, dim):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        #x = torch.matmul(x, self.w_2[:, 0:dim])
        x = self.w_2(x)
        x = self.dropout(x)
        return x

class SA(nn.Module):
    def __init__(self, dropout):
        super(SA, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, pad_mask):
        scale_term = math.sqrt(x.size(-1))
        scores = torch.matmul(x, x.transpose(-2, -1)) / scale_term
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask == 0.0, -1e9)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            scores = scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.matmul(prob, x)

class InrAwaSA(nn.Module):
    def __init__(self, dropout):
        super(InrAwaSA, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, attn_mask, pad_mask, dim_):
        scale_term = math.sqrt(x.size(-1))
        if adj is not None:
            scores = adj.to_dense().unsqueeze(0)
        else:
            scores = torch.matmul(x, x.transpose(-2, -1)) / scale_term
        if pad_mask is not None:
            scores.masked_fill(pad_mask == 0.0, -1e9)               
        if attn_mask is not None:
            scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.matmul(prob, x)


class InrEncoderLayer(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(InrEncoderLayer, self).__init__()
        self.inr_sa_layer = InrAwaSA(dropout)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.sublayer = clones(SubLayerConnect(features), 2)


    def forward(self, x, adj, attn_mask, pad_mask, dim):
        x = self.sublayer[0](x, lambda x:self.inr_sa_layer(x, adj, attn_mask, pad_mask, dim))
        x = self.sublayer[1](x, lambda x:self.ffn_layer(x, dim))
        return x


class InrEncoder(nn.Module):
    def __init__(self, features, layer, depth):
        super(InrEncoder, self).__init__()
        self.layers = clones(layer, depth)

    def forward(self, x, adj, attn_mask, pad_mask, dim):
        residual = x
        for layer in self.layers:
            x = layer(x, adj, attn_mask, pad_mask, dim)
        self.norm = nn.LayerNorm(x.size(-1)).cuda()
        return self.norm(x + residual)            

class MHInrAttn(nn.Module):
    def __init__(self, features, n_head, dropout):
        super(MHInrAttn, self).__init__()
        self.d_h = features // n_head
        self.n_head = n_head
        self.linears = clones(nn.Linear(features, features), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, str_mat, attn_mask):
        b = x.size(0)
        query, key, value = [l(x).view(b, self.h, -1, self.d_h) for l, x in zip(self.linears, x)]
        scale_term = query.size(-1)
        str_mat = str_mat.masked_fill(attn_mask == 0.0, -1e9)
        str_mat = F.softmax(str_mat, dim=-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale_term + str_mat
        if attn_mask is not None:
            scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        x = torch.matmul(prob, value)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h*self.d_h)
        return self.linears[-1](x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, num_of_d):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.num_of_d =num_of_d

    def forward(self, Q, K, V, attn_mask, res_att):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) #+ res_att  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            #scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
            scores.masked_fill(attn_mask == 0.0, -1e9)
        attn = F.softmax(scores, dim=3)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, scores

class MultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k ,d_v, n_heads, num_of_d):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_v).transpose(2, 3)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, res_attn = ScaledDotProductAttention(self.d_k, self.num_of_d)(Q, K, V, attn_mask, res_att)

        context = context.transpose(2, 3).reshape(batch_size, self.num_of_d, -1,
                                                  self.n_heads * self.d_v).squeeze(1)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return nn.LayerNorm(self.d_model).to(self.DEVICE)(output + residual)#, res_attn  #+ residual
    
class Embedding(nn.Module):
    def __init__(self, d_Em):
        super(Embedding, self).__init__()
        self.d_Em = d_Em
        self.node_all = 358 + 307 + 883 + 170
        self.max_n_num = 883
        #self.pos_embed = nn.Embedding(nb_seq, d_Em)
        self.pos_s_embed = nn.Embedding(self.max_n_num, d_Em)
        self.pos_t_embed = nn.Embedding(12, d_Em)
        self.norm_s = nn.LayerNorm(d_Em)
        self.norm_t = nn.LayerNorm(d_Em)
    def forward(self, s_x, t_x):
        batch_size, num_of_vertices, hidden_dim = s_x.shape
        batch_size, t_step, hidden_dim = t_x.shape
        pos_s = torch.arange(num_of_vertices, dtype=torch.long).to(s_x.device)
        pos_s = pos_s.unsqueeze(0).expand(batch_size, num_of_vertices)  
        pos_t = torch.arange(t_step, dtype=torch.long).to(s_x.device)
        pos_t = pos_t.unsqueeze(0).expand(batch_size, t_step) 
        s_x = s_x + self.pos_s_embed(pos_s)
        t_x = t_x + self.pos_t_embed(pos_t)   
        s_x = self.norm_s(s_x)
        t_x = self.norm_t(t_x)
        return s_x, t_x
