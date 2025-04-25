#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from time import time
import shutil
import argparse
import configparser
import yaml
from sklearn.cluster import SpectralClustering
from model.DSTAGNN_my import make_model
from model.meta_stgcn import *
from model.prompt_layers import *
from lib.dataloader import load_weighted_adjacency_matrix,load_weighted_adjacency_matrix2,load_PA
from lib.utils1 import load_graphdata_channel1, get_adjacency_matrix2, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from tensorboardX import SummaryWriter


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(1)


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04_dstagnn.conf', type=str,
                    help="configuration file path")
parser.add_argument('--config_filename', default='configurations/config.yaml', type=str,
                        help='Configuration filename for restoring all models.')
args = parser.parse_args()
# config = configparser.ConfigParser()
# print('Read configuration file: %s' % (args.config))
# config.read(args.config)

with open(args.config_filename) as f:
    config = yaml.safe_load(f)
data_config, training_config = config['Data'], config['Training']

# data_config = config['Data']
# training_config = config['Training']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:1')  #'cpu'     'cuda:0'
print("CUDA:", USE_CUDA, DEVICE)

data_keys = data_config['data_keys']
node_keys = data_config['node_keys']
node_city = data_config['node_city']

adj_merge_all = {}
adj_pa_all = {}
region_cluster = {}
for city in data_keys:
    adj_filename = config[city]['adj_filename']
    graph_signal_matrix_filename = config[city]['graph_signal_matrix_filename']
    stag_filename = config[city]['stag_filename']
    strg_filename = config[city]['strg_filename']
    num_of_vertices = int(config[city]['num_of_vertices'])
    dataset_name = config[city]['dataset_name']
    graph_use = training_config['graph']
    if city == 'PEMS03':
        id_filename = config[city]['id_filename']
    else:
        id_filename = None

    if dataset_name == 'PEMS04' or 'PEMS08' or 'PEMS07' or 'PEMS03':
        adj_mx = get_adjacency_matrix2(adj_filename, num_of_vertices, id_filename=id_filename)
    else:
        adj_mx = load_weighted_adjacency_matrix2(adj_filename, num_of_vertices)
    adj_TMD = load_weighted_adjacency_matrix(stag_filename, num_of_vertices)
    if graph_use =='G':
        adj_merge = adj_mx
    else:
        adj_merge = adj_TMD
    adj_pa = load_PA(strg_filename)
    adj_merge_all[num_of_vertices] = adj_TMD
    # a, b = [], []
    # for i in range(adj_pa.shape[0]):
    #     for j in range(adj_pa.shape[1]):
    #         if(adj_pa[i][j] > 0):
    #             a.append(i)
    #             b.append(j)
    # edge = [a,b]
    # edge_index = torch.tensor(edge, dtype=torch.long).to(DEVICE)
    cluster_num = 50
    lam, H = np.linalg.eig(adj_pa) # H'shape is n*n
    lam = lam.real
    H = H.real
    data = H[:,0:20]
    data=np.squeeze(data)
    idx = SpectralClustering(n_clusters=cluster_num).fit(data).labels_
    idx = torch.tensor(idx, dtype=torch.long)
    region_cluster[num_of_vertices] = idx
    adj_pa = torch.tensor(adj_pa, dtype=torch.long)
    adj_pa_all[num_of_vertices] = adj_pa#edge_index
    


points_per_hour = int(training_config['points_per_hour'])
num_for_predict = int(training_config['num_for_predict'])
len_input = int(training_config['len_input'])


model_name = training_config['model_name']

graph_use = training_config['graph']


learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = 1
d_model = int(training_config['d_model'])
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
num_of_d = in_channels
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
n_heads = int(training_config['n_heads'])
d_k = int(training_config['d_k'])
d_v = d_k

folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
print('folder_dir:', folder_dir)
params_path = os.path.join('myexperiments', dataset_name, folder_dir)



train_loader, val_loader, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
    config, data_config, training_config, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size, data_keys)


net = make_model(DEVICE, num_of_d, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_merge_all,
                 adj_pa_all, adj_merge_all, num_for_predict, len_input, node_keys, d_model, d_k, d_v, n_heads)  # adj_merge adj_pa

net = prompt_transformer(data_config, training_config, DEVICE, d_k, d_v, n_heads, num_of_d, region_cluster).to(device=DEVICE)


def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)

    criterion = nn.SmoothL1Loss().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50, 100, 150, 200], gamma=0.5)  #0.9
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = {358:0, 307:0, 883:0, 170:0}
    best_val_loss = {358:np.inf, 307:np.inf, 883:np.inf, 170:np.inf}

    start_time = time.time()

    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'PEMS04_epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename), strict=False)

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    # train model
    for epoch in range(start_epoch, epochs):
        print('current epoch: ', epoch)
        for node_num in node_keys:
            city_name = node_city[node_num]
            params_filename = os.path.join(params_path, '%s_epoch_%s.params' % (city_name, epoch))
            val_loss = compute_val_loss_mstgcn(net, val_loader, node_num, city_name, criterion, sw, epoch)
            print('val loss', val_loss)
            if val_loss < best_val_loss[node_num]:
                best_val_loss[node_num] = val_loss
                best_epoch[node_num] = epoch
                torch.save(net.state_dict(), params_filename)
                print('%s, best epoch: ' % (city_name), best_epoch[node_num])
                print('%s, best val loss: ' % (city_name), best_val_loss)
                print('%s, save parameters to file: %s' % (city_name, params_filename))

        net.train()  # ensure dropout layers are in train mode
        train_time = time.time()
        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.squeeze(0)
            labels = labels.squeeze(0)

            optimizer.zero_grad()

            outputs = net(encoder_inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

            if global_step % 1000 == 0:

                print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time.time() - start_time))
        #scheduler.step()
        print("train_time", time.time()-train_time)
    print('best epoch:', best_epoch)
    # apply the best model on the test set
    predict_main(best_epoch, test_loader, test_target_tensor, _mean, _std, 'test')


def predict_main(global_step, data_loader, data_target_tensor, _mean, _std, type):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    predict_and_save_results_mstgcn(net, data_config, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type)


if __name__ == "__main__":

    train_main()















