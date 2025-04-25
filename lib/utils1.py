import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvalsh
from scipy.linalg import fractional_matrix_power
from torch.utils.data import TensorDataset
# keshihua
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import seaborn as sns
from time import time
sns.set(font_scale=1.5)

def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

def get_adjacency_matrix2(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                # A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]
    id_mat = np.asmatrix(np.identity(n_vertex))

    # D_row
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # D_com
    #deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))

    # D = D_row as default
    deg_mat = deg_mat_row
    adj_mat = np.asmatrix(adj_mat)

    # wid_A = A + I
    wid_adj_mat = adj_mat + id_mat
    # wid_D = D + I
    wid_deg_mat = deg_mat + id_mat

    # Combinatorial Laplacian
    # L_com = D - A
    com_lap_mat = deg_mat - adj_mat

    if mat_type == 'id_mat':
        return id_mat
    elif mat_type == 'com_lap_mat':
        return com_lap_mat

    if (mat_type == 'sym_normd_lap_mat') or (mat_type == 'wid_sym_normd_lap_mat') or (mat_type == 'hat_sym_normd_lap_mat'):
        deg_mat_inv_sqrt = fractional_matrix_power(deg_mat, -0.5)
        deg_mat_inv_sqrt[np.isinf(deg_mat_inv_sqrt)] = 0.

        wid_deg_mat_inv_sqrt = fractional_matrix_power(wid_deg_mat, -0.5)
        wid_deg_mat_inv_sqrt[np.isinf(wid_deg_mat_inv_sqrt)] = 0.

        # Symmetric normalized Laplacian
        # For SpectraConv
        # To [0, 1]
        # L_sym = D^{-0.5} * L_com * D^{-0.5} = I - D^{-0.5} * A * D^{-0.5}
        sym_normd_lap_mat = id_mat - np.matmul(np.matmul(deg_mat_inv_sqrt, adj_mat), deg_mat_inv_sqrt)

        # For ChebConv
        # From [0, 1] to [-1, 1]
        # wid_L_sym = 2 * L_sym / lambda_max_sym - I
        #sym_max_lambda = max(np.linalg.eigvalsh(sym_normd_lap_mat))
        sym_max_lambda = max(eigvalsh(sym_normd_lap_mat))
        wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / sym_max_lambda - id_mat

        # For GCNConv
        # hat_L_sym = wid_D^{-0.5} * wid_A * wid_D^{-0.5}
        hat_sym_normd_lap_mat = np.matmul(np.matmul(wid_deg_mat_inv_sqrt, wid_adj_mat), wid_deg_mat_inv_sqrt)

        if mat_type == 'sym_normd_lap_mat':
            return sym_normd_lap_mat
        elif mat_type == 'wid_sym_normd_lap_mat':
            return wid_sym_normd_lap_mat
        elif mat_type == 'hat_sym_normd_lap_mat':
            return hat_sym_normd_lap_mat

    elif (mat_type == 'rw_normd_lap_mat') or (mat_type == 'wid_rw_normd_lap_mat') or (mat_type == 'hat_rw_normd_lap_mat'):
        try:
            # There is a small possibility that the degree matrix is a singular matrix.
            deg_mat_inv = np.linalg.inv(deg_mat)
        except:
            print(f'The degree matrix is a singular matrix. Cannot use random walk normalized Laplacian matrix.')
        else:
            deg_mat_inv[np.isinf(deg_mat_inv)] = 0.

        wid_deg_mat_inv = np.linalg.inv(wid_deg_mat)
        wid_deg_mat_inv[np.isinf(wid_deg_mat_inv)] = 0.

        # Random Walk normalized Laplacian
        # For SpectraConv
        # To [0, 1]
        # L_rw = D^{-1} * L_com = I - D^{-1} * A
        rw_normd_lap_mat = id_mat - np.matmul(deg_mat_inv, adj_mat)

        # For ChebConv
        # From [0, 1] to [-1, 1]
        # wid_L_rw = 2 * L_rw / lambda_max_rw - I
        #rw_max_lambda = max(np.linalg.eigvalsh(rw_normd_lap_mat))
        rw_max_lambda = max(eigvalsh(rw_normd_lap_mat))
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / rw_max_lambda - id_mat

        # For GCNConv
        # hat_L_rw = wid_D^{-1} * wid_A
        hat_rw_normd_lap_mat = np.matmul(wid_deg_mat_inv, wid_adj_mat)

        if mat_type == 'rw_normd_lap_mat':
            return rw_normd_lap_mat
        elif mat_type == 'wid_rw_normd_lap_mat':
            return wid_rw_normd_lap_mat
        elif mat_type == 'hat_rw_normd_lap_mat':
            return hat_rw_normd_lap_mat

class traffic_dataset(TensorDataset):
        def __init__(self, config, data_config, training_config, num_of_hours, num_of_days, num_of_weeks, x_state, target_state, device):
            super(traffic_dataset, self).__init__()
            self.config = config
            self.data_config = data_config
            self.training_config = training_config
            self.data_list = np.array(data_config['data_keys'])
            self.node_list = np.array(data_config['node_keys'])
            self.num_of_hours = num_of_hours
            self.num_of_days = num_of_days
            self.num_of_weeks = num_of_weeks
            self.DEVICE = device
            self.file_data = self.load_data(x_state, target_state)

        def load_data(self, x_state, target_state):
            self.x_list, self.y_list = [], []
            self.mean_list, self.std_list = [], []
            batch_size = self.training_config['batch_size']

            for dataset_name in self.data_list:
                graph_signal_matrix_filename = self.config[dataset_name]['graph_signal_matrix_filename']
                file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
                dirpath = os.path.dirname(graph_signal_matrix_filename)
                filename = os.path.join(dirpath,
                                        file + '_r' + str(self.num_of_hours) + '_d' + str(self.num_of_days) + '_w' + str(self.num_of_weeks)) +'_dstagnn'
                print('load file:', filename)
                file_data = np.load(filename + '.npz')


                # city = {358:'PEMS03', 307:'PEMS04', 883:'PEMS07', 170:'PEMS08'}

                # train_x_all = file_data[x_state]  # (15711, 358, 12) (10181, 307, 3, 12)
                # train_target_all = file_data[target_state]
                # print(train_x_all.shape, train_target_all.shape, len(train_x_all))
                # node_num = train_x_all.shape[1]

                # for k in range(node_num):
                # #k = 105
                #     pred = []
                #     true = []
                #     for i in range(24*10):
                #         datapred = train_x_all[864+(12*i),k,:]
                #         datatrue = train_target_all[864+(12*i),k,:]
                #         pred.append(datapred)
                #         true.append(datatrue)
                #     pred = np.array(pred).flatten().tolist()
                #     true = np.array(true).flatten().tolist()
                #     plt.figure()
                #     plt.plot(pred,color = 'red',label = 'Prediction')
                #     plt.plot(true,color = 'grey',label = 'Truth')
                #     plt.legend(loc="upper right")
                #     plt.title('Test prediction vs Target')
                #     plt.savefig('./figure/train-{}-{}.png'.format(city[node_num], k))
                #     plt.show()


                # # Plot the test prediction vs target（optional)
                # plt.figure(figsize=(10, 20))

                # #for k in range(node_num):
                # k = 49
                # #plt.subplot(node_num, 1, k + 1)
                # for j in range(len(train_x_all)//288):
                #     c, d = [], []
                #     for t in range(24):
                #         time = j*288 + t*12
                #         for i in range(12):
                #             c.append(train_x_all[time, k, i])
                #             d.append(train_target_all[time, k, i])
                #     #plt.plot(range(1 + time, 12*24 + 1 + time), c, c='b')
                #     plt.plot(range(0, 12*24), d, c='r')
                    # plt.title('Test prediction vs Target')
                    # plt.savefig('./figure/test_results.png')



                train_x_all = file_data[x_state]
                train_target_all = file_data[target_state]
                for i in range(file_data[x_state].shape[0] // batch_size):
                    train_x = train_x_all[i*batch_size:(i+1)*batch_size, :, 0:1, :]
                    train_target = train_target_all[i*batch_size:(i+1)*batch_size]  # (10181, 307, 12)
                    train_x = torch.from_numpy(train_x).type(torch.FloatTensor).to(self.DEVICE)
                    train_target = torch.from_numpy(train_target).type(torch.FloatTensor).to(self.DEVICE)
                    self.x_list.append(train_x)
                    self.y_list.append(train_target)


                train_x = train_x_all[(i+1)*batch_size:, :, 0:1, :]
                train_target = train_target_all[(i+1)*batch_size:]  # (10181, 307, 12)
                train_x = torch.from_numpy(train_x).type(torch.FloatTensor).to(self.DEVICE)
                train_target = torch.from_numpy(train_target).type(torch.FloatTensor).to(self.DEVICE)
                self.x_list.append(train_x)
                self.y_list.append(train_target)
            # self.x_list = torch.from_numpy(np.array(self.x_list)).type(torch.FloatTensor).to(self.DEVICE)
            # self.y_list = torch.from_numpy(np.array(self.y_list)).type(torch.FloatTensor).to(self.DEVICE)
            return file_data
    
        def __getitem__(self, index):
            x = self.x_list[index]
            target = self.y_list[index]
            return x, target
        def get_target_tensor(self):
            target_tensor = {}
            for index in range(len(self.y_list)):
                node_num = self.y_list[index].shape[1]
                if node_num not in target_tensor:
                    target_tensor[node_num] = []
                target_tensor[node_num].append(self.y_list[index])
            for node_num in target_tensor.keys():
                target_tensor[node_num] = torch.cat(target_tensor[node_num], dim=0)
            return target_tensor
        
        def __len__(self):
            data_length = len(self.x_list)
            return data_length

def load_graphdata_channel1(config, data_config, training_config, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, data_keys, shuffle=True):
    '''
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
    这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    # ------- train_loader -------
    train_dataset = traffic_dataset(config, data_config, training_config, num_of_hours, num_of_days, num_of_weeks, 'train_x', 'train_target', DEVICE)  # (B, N, F, T)  # (B, N, T)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=shuffle)

    # ------- val_loader -------
    val_dataset = traffic_dataset(config, data_config, training_config, num_of_hours, num_of_days, num_of_weeks, 'val_x', 'val_target', DEVICE)  # (B, N, F, T)  # (B, N, T)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ------- test_loader -------
    test_dataset = traffic_dataset(config, data_config, training_config, num_of_hours, num_of_days, num_of_weeks, 'test_x', 'test_target', DEVICE)  # (B, N, F, T)  # (B, N, T)
    test_target_tensor = test_dataset.get_target_tensor()

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # # print
    # print('train:', train_x_tensor.size(), train_target_tensor.size())
    # print('val:', val_x_tensor.size(), val_target_tensor.size())
    # print('test:', test_x_tensor.size(), test_target_tensor.size())
    mean_dict, std_dict = {}, {}
    data_list = np.array(data_config['data_keys'])
    for dataset_name in data_list:
        graph_signal_matrix_filename = config[dataset_name]['graph_signal_matrix_filename']
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath,
                                file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) +'_dstagnn'
        print('load file:', filename)
        file_data = np.load(filename + '.npz')
        mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
        std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)
        mean_dict[config[dataset_name]['num_of_vertices']] = mean
        std_dict[config[dataset_name]['num_of_vertices']] = std
        
    return train_loader, val_loader, test_loader, test_target_tensor, mean_dict, std_dict


def compute_val_loss_mstgcn(net, val_loader, node_num, city_name, criterion, sw, epoch, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode
    val_time = time()
    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.squeeze(0)
            labels = labels.squeeze(0)
            if encoder_inputs.shape[1] != node_num:
                continue
            outputs = net(encoder_inputs)
            loss = criterion(outputs, labels)  # 计算误差
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break
        print("val_time:", time()-val_time)
        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('%s_validation_loss' % (city_name), validation_loss, epoch)
    return validation_loss


def evaluate_on_test_mstgcn(net, test_loader, test_target_tensor, sw, epoch, _mean, _std):
    '''
    for rnn, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.

    :param net: model
    :param test_loader: torch.utils.data.utils.DataLoader
    :param test_target_tensor: torch.tensor (B, N_nodes, T_output, out_feature)=(B, N_nodes, T_output, 1)
    :param sw:
    :param epoch: int, current epoch
    :param _mean: (1, 1, 3(features), 1)
    :param _std: (1, 1, 3(features), 1)
    '''

    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        test_loader_length = len(test_loader)

        test_target_tensor = test_target_tensor.cpu().numpy()

        prediction = []  # 存储所有batch的output

        for batch_index, batch_data in enumerate(test_loader):

            encoder_inputs, labels = batch_data

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert test_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(test_target_tensor[:, :, i], prediction[:, :, i])
            rmse = mean_squared_error(test_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            mape = masked_mape_np(test_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            print()
            if sw:
                sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
                sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
                sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)


def predict_and_save_results_mstgcn(net, data_config, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type):
    '''

    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    data_list = data_config['data_keys']
    node_city = data_config['node_city']
    node_list = data_config['node_keys']
    for node_num in node_list:
        city_name = node_city[node_num] 
        data_target_tensor[node_num] = data_target_tensor[node_num].cpu().numpy()
        params_filename = os.path.join(params_path, '%s_epoch_%s.params' % (city_name, global_step[node_num]))
        print('load weight from:', params_filename)
        net.load_state_dict(torch.load(params_filename), strict=False)
        results = {}
        net.train(False)  # ensure dropout layers are in test mode

        with torch.no_grad():
            loader_length = len(data_loader)  # nb of batch

            prediction = []  # 存储所有batch的output

            input = []  # 存储所有batch的input

            for batch_index, batch_data in enumerate(data_loader):

                encoder_inputs, labels = batch_data
                encoder_inputs = encoder_inputs.squeeze(0)
                node_num_loader = encoder_inputs.shape[1]
                if node_num_loader != node_num:
                    continue
                input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)

                outputs, cl_loss = net(encoder_inputs)

                prediction.append(outputs.detach().cpu().numpy())

                if batch_index % 100 == 0:
                    print('%s predicting data set batch %s / %s' % (city_name, batch_index + 1, loader_length))

            input = np.concatenate(input, 0)
            input = re_normalization(input, _mean[node_num], _std[node_num])
            prediction = np.concatenate(prediction, 0)  # (batch, T', 1)

            print('input:', input.shape)
            print('prediction:', prediction.shape)
            print('data_target_tensor:', data_target_tensor[node_num].shape)
            output_filename = os.path.join(params_path, 'output_%s__epoch_%s_%s' % (city_name, global_step[node_num], type))
            np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor[node_num])

            # 计算误差
            excel_list = []
            prediction_length = prediction.shape[2]

            for i in range(prediction_length):
                assert data_target_tensor[node_num].shape[0] == prediction.shape[0]
                print('current epoch: %s, predict %s-th point' % (global_step[node_num], i+1))
                mae = mean_absolute_error(data_target_tensor[node_num][:, :, i], prediction[:, :, i])
                rmse = mean_squared_error(data_target_tensor[node_num][:, :, i], prediction[:, :, i]) ** 0.5
                mape = masked_mape_np(data_target_tensor[node_num][:, :, i], prediction[:, :, i], 0)
                print('%s, MAE: %.2f' % (city_name, mae))
                print('%s, RMSE: %.2f' % (city_name, rmse))
                print('%s, MAPE: %.2f' % (city_name, mape))
                excel_list.extend([mae, rmse, mape])

            # print overall results
            mae = mean_absolute_error(data_target_tensor[node_num].reshape(-1, 1), prediction.reshape(-1, 1))
            rmse = mean_squared_error(data_target_tensor[node_num].reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
            mape = masked_mape_np(data_target_tensor[node_num].reshape(-1, 1), prediction.reshape(-1, 1), 0)
            print('%s, all MAE: %.2f' % (city_name, mae))
            print('%s, all RMSE: %.2f' % (city_name, rmse))
            print('%s, all MAPE: %.2f' % (city_name, mape))
            excel_list.extend([mae, rmse, mape])
            print(excel_list)





