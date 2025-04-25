import torch
import torch.nn as nn
import torch.nn.functional as F

class prompt_layer_weighted_node(nn.Module):   #[358, 307, 883, 170]
    def __init__(self, input_dim, training_config, layer_num=8):
        super(prompt_layer_weighted_node, self).__init__()
        self.n_node = 4
        self.node_all = 358 + 307 + 883 + 170
        max_n_num = 883
        self.region_num = 50 * 4
        self.node_num = torch.tensor([358, 307, 883, 170])   #[358, 307, 883, 170]
        self.weight_s = torch.nn.Embedding(self.node_all * layer_num, input_dim)
        self.weight_t = torch.nn.Embedding(self.n_node * 12 * layer_num, input_dim)
        self.wcity_s = torch.nn.Embedding(self.region_num * layer_num, input_dim)
        #self.wcity_t = torch.nn.Embedding(self.n_node * 12 * layer_num, input_dim)
        # self.weight_s = torch.nn.Embedding(4, input_dim)
        # self.weight_t = torch.nn.Embedding(4, input_dim)
        self.linear_spatial = nn.Linear(2*training_config['points_per_hour'], input_dim)
        self.linear_temporal = torch.nn.Parameter(torch.Tensor(max_n_num * 2, input_dim))

        self.linear_weight_spatial = torch.nn.Parameter(torch.Tensor(input_dim, max_n_num * input_dim))
        self.linear_weight_temporal = nn.Linear(input_dim, input_dim)
        self.input_dim = input_dim
        self.reset_parameters()
    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight_s)
        # torch.nn.init.xavier_uniform_(self.weight_t)
        #torch.nn.init.xavier_uniform_(self.linear_spatial)
        torch.nn.init.xavier_uniform_(self.linear_temporal)
        torch.nn.init.xavier_uniform_(self.linear_weight_spatial)
        #torch.nn.init.xavier_uniform_(self.linear_weight_temporal)
    def forward(self, embedding_s, embedding_t, city, region, layer_num):
        # if layer_num == 0:
        #     #embedding -> [btach_size, node_num, his_len, hidden_dim]
        #     batch_size, node_num, seq_len, hidden_dim = embedding_s.shape
        #     #city = torch.LongTensor(city).repeat(batch_size, node_num, seq_len).cuda()
        #     #embedding_s = self.linear_spatial(embedding)
        #     #embedding_s => [batch_size, N, T*d]
        #     embedding_s = embedding_s.reshape(batch_size, node_num, seq_len*hidden_dim)
        #     embedding_s = self.linear_spatial(embedding_s)
        #     #embedding_t => [batch_size, T, N*d]
        #     embedding_t = embedding_t.transpose(1,2).reshape(batch_size, seq_len, node_num*hidden_dim)
        #     weight_t_ = self.linear_temporal[0:node_num*hidden_dim].unsqueeze(0)
        #     embedding_t = torch.matmul(embedding_t, weight_t_)
        
        # city_s = torch.arange(self.node_num[:city].sum(dim=0).item()  if c != 0 else 0, self.node_num[:city+1].sum(dim=0).item())
        # city_s = city_s.unsqueeze(0).repeat(batch_size, 1).cuda()  #[b, N]
    
    #provide a prompt for each node and each timestamp with each layer
        city_s = [torch.arange(self.node_num[:c].sum(dim=0).item()  if c != 0 else 0, self.node_num[:(c+1)].sum(dim=0).item()) for c in city]
        city_s = torch.stack(city_s, dim=0).to(embedding_s.device)
        city_s = city_s.add(self.node_all * layer_num)
        
        region = region + (50 * city[0]).to(embedding_s.device)
        region = region.add(self.region_num * layer_num)

        city_t = [torch.arange(12*c, 12*(c+1)) for c in city]
        city_t = torch.stack(city_t, dim=0).to(embedding_s.device)
        city_t = city_t.add(12 * layer_num) 

        weight_t = self.weight_t(city_t.to(embedding_s.device))
        weight_s = self.weight_s(city_s.to(embedding_s.device))
        wcity_s = self.wcity_s(region.to(embedding_s.device))
    
    # #provide a prompt for each node and each timestamp
    #     city_s = [torch.arange(self.node_num[:c].sum(dim=0).item() if c != 0 else 0, self.node_num[:c+1].sum(dim=0).item()) for c in city]
    #     city_s = torch.stack(city_s, dim=0).cuda()

    #     city_t = [torch.arange(12*c, 12*(c+1)) for c in city]
    #     city_t = torch.stack(city_t, dim=0).cuda()

    #     weight_t = self.weight_t(city_t.cuda())
    #     weight_s = self.weight_s(city_s.cuda())


    # #provide a prompt for a city
    #     batch_size, node_num, _ = embedding_s.shape
    #     city = city.add(self.n_node * layer_num)
    #     weight_t = self.weight_t(city.cuda()).unsqueeze(1).repeat(1,embedding_t.size(1),1)
    #     weight_s = self.weight_s(city.cuda()).unsqueeze(1).repeat(1,embedding_s.size(1),1)
        
    #     weight_t_ = self.linear_weight_spatial[0:self.input_dim, 0:self.input_dim].unsqueeze(0).repeat(batch_size,1,1)
    #     weight_t = torch.bmm(weight_t, weight_t_)
    #     weight_s = self.linear_weight_temporal(weight_s)
        
        embedding_t = torch.cat([embedding_t, weight_t], dim=-1)
        embedding_s = torch.cat([embedding_s, weight_s], dim=-1) 
        #embedding_s = torch.cat([embedding_s, wcity_s], dim=-1)
        #embedding_t = embedding_t + weight_t     
        #embedding_s = embedding_s + weight_s

        #prompt: mean
        #prompt_result = embedding.sum(dim=1)

        return embedding_s, embedding_t



