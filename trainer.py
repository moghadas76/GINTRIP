"""
Our trainer(ProtoTraffic)
TODO:
    - Add Fidelity and Sparsity (*)
    - Draw Sub-graphs(*)
    - Hyperparameter Tuning(Lambdas)(*)
    - Table for comparision with STExplainer
    - Add Prototype Projection monte carlo tree search(**Important)(to be test) (*)
    - Add Merge Prototype(**Important)
    - Subgraph visualization G-sub(*)
    - (S) For each epoch test inference
        - Tensorboard test
    - (S) Regression only
    - changigng in args.clip (***********************important)
    - self.predefined_A --> for connectivity_loss (**important)
    - congestion in every condition should be done ---> in  location :test 2
    - discovery on lambas binary-wise ---> optuna ?


After idea:
    - (A) Scheduling of lambdas(**important)
        - Schadule Of LR
    - Top_k based on the k Tensorboard

    - Diffrent backbones
    - Fidelity-
    - Subgraph Augmentation
        - Subgraph classification based on different scales(Assign sudo-labels foreach subgraph)
    - (A) Balance / Imbalance(currently 0.66 for the class 1)

Optuna:
    - Optuna Dashboard & Hub(*)
    - Optunal on Number of protos(14)
    - Optunal on test error on final phase(Important)
    - dim. of prototype --> 128 ?
    - Optuna on Hidden dimension(**importnant)
    - Annealing(optional)

After Submission:
    - Fix after iteration cuda memory
    - Increase epochs
    - Multi-gpu

Optional:
    - integrating domain knowledge into prototype learning by imposing constraints on subgraphs.
    - Connectitivy loss for local subgraph detection(Maybe? Exist or Remove?)

Doings:
    - Topk is buggy
Dataset Pems4(Pems04)
"""
import os
import json
import time
import math
from typing import TypeVar
import torch
import pickle
from tqdm import tqdm
import joblib
import optuna
import inspect
import argparse
import datetime
import subprocess
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch.nn.functional as F
import torch.optim as optim

from scipy.sparse import linalg
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_networkx
from my_mcts import mcts

from proto_join import join_prototypes_by_activations
from models.traffic import ProtoGtnet

from optuna.pruners import SuccessiveHalvingPruner
from cov_weighting.covweighting_loss import CoVWeightingLoss

from PCGrad.pcgrad import PCGrad

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from trainer import Trainer


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def shuffle_and_sample(self, indices: list):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs[indices[0]:indices[1], ...]
        self.ys = ys[indices[0]:indices[1], ...]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def dataloader_to_generator(dataloader):
    for data in dataloader:
        yield data


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, train_split_flag=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    min_val = data['x_train'][..., 0].min()
    # Data format
    congestion = None
    for category in ['train', 'val', 'test']:
        if category == 'train':
            quantiles = np.quantile(data['x_' + category][..., 0], 0.10, axis=0)
            quantiles_bk = np.quantile(data['x_' + category][..., 0], 0.15, axis=0)
            congestion = np.where(quantiles == 0.0, quantiles_bk, quantiles)[0, :]
            quantiles_bk = np.quantile(data['x_' + category][..., 0], 0.25, axis=0)
            congestion = np.where(congestion == 0.0, quantiles_bk, congestion)[0, :]
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    # if train_split_flag:
    #     dataloader = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    #     dataloader.shuffle_and_sample([0, train_split_flag])
    #     data['train_loader'] = dataloader
    #     # data['train_loader'] = itertools.islice(dataloader_to_generator(dataloader), 0, train_split_flag)
    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    data['min_val'] = min_val
    data["congestion"] = congestion
    return data


def load_graphdata_channel1(
        graph_signal_matrix_filename,
        num_of_hours,
        num_of_days,
        num_of_weeks,
        DEVICE,
        batch_size,
        shuffle=True
):
    '''
    Inspired from https://github.com/SYLan2019/DSTAGNN/
    :param graph_signal_matrix_f400ilename: str
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
    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(
                                num_of_weeks)) + '_dstagnn'

    print('load file:', filename)

    file_data = np.load(filename + '.npz')
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']

    mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    quantiles = np.quantile(torch.from_numpy(train_x)[..., 0], 0.10, axis=0)
    quantiles_bk = np.quantile(torch.from_numpy(train_x)[..., 0], 0.15, axis=0)
    congestion = np.where(quantiles == 0.0, quantiles_bk, quantiles)[0, :]
    quantiles_bk = np.quantile(torch.from_numpy(train_x)[..., 0], 0.25, axis=0)
    congestion = np.where(congestion == 0.0, quantiles_bk, congestion)[0, :]
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False)  #TODO: shuffle=False ----> shuffle=shuffle

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)  #TODO: shuffle=False ----> shuffle=shuffle
    min_val = train_x.min()
    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_x_tensor, train_loader, train_target_tensor, val_x_tensor, val_loader, val_target_tensor, test_x_tensor, test_loader, test_target_tensor, mean, std, congestion, min_val


def masked_mse(preds, labels, *args, null_val=0.0, **kwargs ):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, *args, null_val=0.0, **kwargs ):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


UNNORMALIZE = TypeVar('UNNORMALIZE')


def masked_mae(preds: UNNORMALIZE, labels: UNNORMALIZE, *args, null_val=0.0, **kwargs):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def inverse_transform(normalized_value, scaler, min_val):
    prev_mean = scaler.mean
    scaler.mean = min_val
    unnormalized = scaler.inverse_transform(normalized_value)
    scaler.mean = prev_mean
    return unnormalized


def ib_masked_mae(preds: UNNORMALIZE, labels, *args, null_val=0.0, iner_coeff=1.0):
    """
    torch.Size([64, 1, 207, 1])
               [ B, P, N  , T]
    """
    # TODO: preds.dtype != labels.dtype ---> check please !!!
    # TODO: here we have not any nan value in this dataset.
    epsilon = torch.tensor(1e-9).to(preds.device)
    if np.isnan(null_val):
        mask_ = ~torch.isnan(labels)
    else:
        mask_ = (labels != null_val)
    mask = mask_.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # preds = args[0].transform(preds) # Hint: it is not a normalozation. Just for canceel out the input that is UNNORMALIZE
    labels = args[0].transform(labels) - args[0].transform(torch.tensor(args[1]).to(labels.device))
    # TODO: cumpute min on all data set

    # labels = labels[mask_] 
    """ 
    TODO: be positive preds ---> idea : pass preds(before un_normalize) to relu. ---> after that we need pass to un_normalize ? yes or not ?
    resolve 0 problem (both in support and zeros arrised from torch.isnan (missing data)) ---> idea : compute component just on nonzero_support.
    negetive problem in GT ---> idea : compute min of GT (on all train data) and substract it from all GT.
    """
    # preds = preds[mask_]

    preds_mask = (preds >= 0.0) #TODO (important) --> we have used a F.relu at the end of the nettwork,
    #TODO but why somtimes the "preds" get a non-positive value ?!!!!!!

    alpha_ = labels.sum(-1, keepdim=True)
    beta_ = preds.sum(-1, keepdim=True)

    alpha = (labels * preds_mask).sum(-1, keepdim=True)
    beta = (preds * preds_mask).sum(-1, keepdim=True)

    norm_labels = labels / (alpha + epsilon)
    norm_preds = preds / (beta + epsilon)

    norm_preds_temp = torch.where(norm_preds >= 0.0, norm_preds, torch.inf)
    lambda_coff = norm_preds_temp.min(-1)[0] + epsilon

    # lambda_coff.detach_()

    # if lambda_coff == torch.inf:
    #     return torch.tensor(0.0, device=norm_preds.device, dtype=norm_preds.dtype)


    # mask_pred = torch.where(norm_preds == 0.0, torch.zeros_like(norm_preds), torch.ones_like(norm_preds))
    # mae = (torch.abs((norm_labels - norm_preds) * mask_pred * torch.sqrt(mask)).sum(-1)) ** 2

    # mask_pred = (norm_preds != 0.0).float()
    mae = (torch.abs((norm_labels - norm_preds) * preds_mask).sum(-1)) ** 2

    # lambda_coff_mask = (lambda_coff != torch.inf)
    # scaled_mae = mae / lambda_coff
    scaled_mae = mae

    scaled_mae = scaled_mae + iner_coeff * ((alpha_ - beta_).squeeze(-1)) ** 2
    final_loss = torch.mean(0.5 * scaled_mae)

    # if (final_loss > 100000.0) or (final_loss == torch.nan):
    #     print("@@@@@@@@@", final_loss, "@@@@@@@")
    #     # return torch.tensor(0.0, device=norm_preds.device, dtype=norm_preds.dtype)
    #     scaled_mae = mae + iner_coeff * ((alpha_ - beta_).squeeze(-1)) ** 2
    #     final_loss = torch.mean(scaled_mae)

    return final_loss

    # loss = loss * mask # torch.Size([64, 1, 207, 1])
    # loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # return torch.mean(loss)


def masked_mape(preds, labels, *args, null_val=0.0, **kwargs):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / (labels + torch.finfo(labels.dtype).eps)#TODO

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z = torch.tensor((x - mean) / std, dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Trainer():
    def __init__(self,
                 epoch,
                 model,
                 lrate,
                 wdecay,
                 clip,
                 step_size,
                 seq_out_len,
                 scaler,
                 min_val,
                 device,
                 cl=True,
                 congestion_th=None,
                 writer: SummaryWriter = None,
                 lamda_1=None,
                 lamda_2=None,
                 lamda_3=None,
                 lamda_4=None,
                 mi_loss=False,
                 loss_mse=False,
                 graph_reg=False
                 ):
        self.scaler = scaler
        self.min_val = min_val
        self.epoch = epoch
        self.model = model
        self.model.to(device)

        if args.PCGrad:
            self.optimizer = PCGrad(getattr(optim, args.optimizer)(self.model.parameters(),
                                    weight_decay=wdecay))
        else:
            self.optimizer = getattr(optim, args.optimizer)(self.model.parameters(), weight_decay=wdecay)  #TODO: Dose LearningRateScheduler need?

        if args.scheduler:
            if args.scheduler == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch)
            elif args.scheduler == "exponential":
                self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
            elif args.scheduler == "ReduceLROnPlateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
            else:
                raise Exception("Sorry, args.scheduler has not correct value")

        if args.swa:
            self.swa_model = optim.swa_utils.AveragedModel(self.model)
            self.swa_start = 200
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch)
            # self.swa_scheduler = optim.swa_utils.SWALR(self.optimizer, anneal_strategy="linear",
            #                                           anneal_epochs=5, swa_lr=0.05)
            self.swa_scheduler = optim.swa_utils.SWALR(self.optimizer, swa_lr=0.05)

        self.mi_loss = mi_loss
        self.loss_mse = loss_mse
        self.graph = graph_reg
        if self.mi_loss:
            self.loss = ib_masked_mae
        elif self.loss_mse: #TODO
            loss_map = {
                0: masked_mse,
                1: masked_rmse,
                2: masked_mape
            }
            # self.loss = masked_mse #TODO
            # self.loss = masked_rmse #TODO
            self.loss = loss_map[self.loss_mse] #TODO
        else:
            self.loss = masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.writer = writer
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl
        self.congestion_th = torch.from_numpy(congestion_th).to(device)
        self.lambdas = [lamda_1, lamda_2, lamda_3, lamda_4]
        self.criterion = CoVWeightingLoss(args)
        # Record the mean weights for an epoch.
        self.mean_weights = [0.0 for _ in range(self.criterion.alphas.shape[0])]

    def train(self, input, real_val, idx=None, lambdas: list = None, train_loader_len=None):
        self.lambdas = lambdas
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, idx=idx)
        (logits, probs, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, min_distance,
         congestion, adp, importance, x_comp, sparsity, sampled_adj) = output
        output = logits
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        # if not self.mi_loss or self.loss_mse: #TODO
        if self.mi_loss:
            predict = output
        else:
            predict = self.scaler.inverse_transform(output)

        # congestion_th = torch.from_numpy(self.congestion_th).to(real.device)
        congestion_gt = (real[:, :, :].mean(dim=-1).squeeze(0) < self.congestion_th.to(real.device)).squeeze(1).to(
            torch.long)

        try:
            congestion_loss = F.cross_entropy(congestion.transpose(1, 2), congestion_gt)
        except IndexError:
            congestion_loss = torch.tensor(0.0).to(real.device)
        if self.iter % self.step == 0 and self.task_level <= self.seq_out_len:
            self.task_level += 1
        if self.cl:
            loss_reg = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level],
                                 self.scaler, self.min_val, null_val=0.0, iner_coeff=1.0)
        else:
            loss_reg = self.loss(predict, real, self.scaler, self.min_val, null_val=0.0, iner_coeff=1.0)
        loss_tmp = loss_reg.clone()
        np.savez('predict.npz', np.array(probs))

        ###############################################################

        if self.graph:
            graph_loss = torch.nn.MSELoss()(self.model.predefined_A +
                                            (torch.eye(self.model.predefined_A[1].shape[0]).to(
                                                self.model.predefined_A.device).repeat(sampled_adj.shape[0], 1, 1)),
                                            sampled_adj)
            self.writer.add_scalar('graph_reconstruction/train', graph_loss.mean(), self.iter)

        if args.CoVWeightingLoss:
            self.criterion.to_train()

            if args.CoVWeightingLoss_custom:

                losses_list = [prototype_pred_loss.mean(), connectivity_loss.mean(),
                               KL_Loss.mean(), congestion_loss.mean()]
                losses_list_name = ["prototype_pred_loss.mean()", "connectivity_loss.mean()",
                                    "KL_Loss.mean()", "congestion_loss.mean()"]



            else:
                losses_list = [loss_reg.mean(), prototype_pred_loss.mean(), connectivity_loss.mean(),
                               KL_Loss.mean(), congestion_loss.mean()]
                losses_list_name = ["loss_reg.mean()", "prototype_pred_loss.mean()", "connectivity_loss.mean()",
                                    "KL_Loss.mean()", "congestion_loss.mean()"]

            if self.graph:
                losses_list.append(graph_loss.mean())
                losses_list_name.append("graph_loss.mean()")

            if args.CoVWeightingLoss_custom:
                loss = self.criterion.forward(losses_list) + 1.0 * loss_reg.mean()
            else:
                loss = self.criterion.forward(losses_list)

            for i, weight in enumerate(self.criterion.alphas):
                self.mean_weights[i] += (
                            weight.item() / train_loader_len)  #TODO: dose not assign to zero in first of each epoch?

                # self.writer.add_scalar(f"mean_weights/{losses_list_name[i]}", self.mean_weights[i], self.iter)
                self.writer.add_scalar(f"weights/{losses_list_name[i]}", weight.item(), self.iter)  #TODO
                # print(weight.item())
                # breakpoint()
        elif args.PCGrad:

            losses_list = [loss_reg.mean(), prototype_pred_loss.mean(), connectivity_loss.mean(),
                           KL_Loss.mean(), congestion_loss.mean()]

            if self.graph:
                losses_list.append(graph_loss.mean())


            loss = sum(losses_list)


        else:
            # Coeffs adopted from GIB code
            # loss = loss.mean() + 0.01 * prototype_pred_loss.mean() + 5 *connectivity_loss.mean() + 0.0001 * KL_Loss.mean() + 0.0001 * congestion_loss.mean()
            loss = loss_reg.mean() + self.lambdas[0] * prototype_pred_loss.mean()
            + self.lambdas[1] * connectivity_loss.mean()
            + self.lambdas[2] * KL_Loss.mean() \
            + self.lambdas[3] * congestion_loss.mean()
            if self.graph:
                loss = loss + self.lambdas[3] * graph_loss.mean()  # TODO: self.lambdas[3] ?? OR self.lambdas[4] ?
                #TODO: add self.graph to Test section (both test 1 and 2)
                #TODO: add self.graph to Test section (both test 1 and 2)

            # loss = 0.01 * prototype_pred_loss + 5 *connectivity_loss + 0.0001 * KL_Loss.mean()

        ##############################################################
        if args.PCGrad:
            self.optimizer.pc_backward(losses_list)
            # for p in self.model.parameters():
            #     print("#################", p.grad)
        else:
            loss.backward()
        fidelity_plus_modif = torch.abs(output - x_comp[0]).mean()
        fidelity_norm = torch.norm(output - x_comp[0])
        fidelity_plus_topk = torch.abs(output - x_comp[1]).mean()

        self.writer.add_scalar('Fidelity+/train', fidelity_plus_modif, self.iter)
        self.writer.add_scalar('Fidelity_Norm/train', fidelity_norm, self.iter)
        self.writer.add_scalar('Fidelity_topk/train', fidelity_plus_topk, self.iter)
        self.writer.add_scalar('Sparsity/train', sparsity, self.iter)
        self.writer.add_scalar('congestion_loss/train', congestion_loss.mean().detach().item(), self.iter)
        self.writer.add_scalar('KL_Loss/train', KL_Loss.mean().detach().item(), self.iter)
        self.writer.add_scalar('prototype_pred_loss/train', prototype_pred_loss.mean().detach().item(), self.iter)
        self.writer.add_scalar('connectivity_loss/train', connectivity_loss.mean().detach().item(), self.iter)
        self.writer.add_scalar('reg_loss/train', loss_tmp.mean().detach().item(), self.iter)

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        # if self.mi_loss or self.loss_mse:
        if self.mi_loss:
            predict = inverse_transform(predict, self.scaler, self.min_val)
        mae = masked_mae(predict, real, 0.0, scalar=self.scaler).item()
        self.writer.add_scalar('MAE/train', mae, self.iter)
        mape = masked_mape(predict, real, 0.0).item()
        rmse = masked_rmse(predict, real, 0.0).item()
        self.iter += 1
        return loss.item(), mape, rmse, adp

    @staticmethod
    def visualize(input, adp, probs,testy, preds, fidelity_plus_modif, congestion, matrix, iter_num, importance):
        adj, _ = dense_to_sparse(adp)
        data_geometric = Data(input, adj)
        G = to_networkx(data_geometric, to_undirected=True)
        # subgraph = G.subgraph(probs[0])
        subgraphs = []
        based_time = pd.Timestamp("2018-01-01").to_period(freq="5T") + 10181 + 3394 + iter_num
        for i in range(64):
            subgraphs.append([])
        for (i, j) in probs:
            subgraphs[i].append(j)
        # for k in [5, 10, 15, 20]:
        for k in [15]:
            k_subgraph_node_list = importance[...,0].topk(int((k/100)*307))[1]
            for i, subgraph_list in enumerate(subgraphs):
                # subgraph = G.subgraph(subgraph_list)
                k_subgraph = G.subgraph(k_subgraph_node_list[i].tolist())
                plt.figure(figsize=(20, 15), dpi=300)  # Increased figure size and DPI
                ax1 = plt.subplot(121)
                pos = nx.spring_layout(G, k=0.5, iterations=50)
                # nx.draw(G,pos,
                #     with_labels=True, 
                #     node_color='lightblue',
                #     node_size=90, 
                #     font_size=8, 
                #     edge_color='gray',  # Change edge color to gray
                #     width=0.1,  # Make edges thinner
                #     # alpha=0.3
                # )
                # Create a color map
                # cmap = plt.cm.viridis  # You can change this to other colormaps like 'plasma', 'inferno', etc.
                # node_colors = cmap(importance[...,0][i].cpu().numpy())

                # Draw the graph
                # pos = nx.spring_layout(G)
                # nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=300)

                # Add a colorbar
                # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(importance[...,0][i]), vmax=max(importance[...,0][i])))
                # sm.set_array([])
                # cbar = plt.colorbar(sm)
                # cbar.set_label('Node Degree')

                nx.draw(G,pos,
                    with_labels=True, 
                    node_color='lightblue',
                    node_size=90, 
                    font_size=8, 
                    edge_color='gray',  # Change edge color to gray
                    width=0.1,  # Make edges thinner
                    # alpha=0.3
                )
                nx.draw(k_subgraph,pos,
                    with_labels=True, 
                    node_color='green',
                    node_size=90, 
                    font_size=8, 
                    edge_color='red',  # Change edge color to gray
                    width=2,  # Make edges thinner
                    # alpha=0.3
                )

                plt.title(f"{k}- Subgraph")
                plt.tight_layout()
                plt.savefig(f"./results/log/2023y10m24d/Subgraph_{i}_{k}.png")
                plt.close()
                ground_truth = testy[i,0, k_subgraph_node_list[i].tolist(),:].cpu().numpy()
                predictions = preds[i,0, k_subgraph_node_list[i].tolist(),:].cpu().numpy()
				# Temperature subplot
                plt.subplot(2, 1, 1)
                ind = pd.period_range(start=based_time, periods=12, freq='5T').to_timestamp()
                for node in range(int((k/100)*307)):
                    plt.figure(figsize=(12, 8))
                    plt.plot(ind, ground_truth[node], label=f'Node_{node}')
                    plt.plot(ind, predictions[node], label=f'Prediction_{node}')
                # plt.plot(ind, ground_truth, label='Ground Truth')
                    plt.title(f'Prediction for batch{iter_num}_{i} with {k}% of nodes')
                    plt.legend()
                    plt.savefig(f"./results/log/2023y10m24d/sub_{i}_{k}_{node}.png")
                    plt.close()
            return
                # df = nx.to_pandas_edgelist(subgraph)
    #         df["label"] = [1]*df.shape[0]
    #         df["weight"] = [1]*df.shape[0]
    # # Output json of the graph.
    #         from web import d3
    #         # breakpoint()
        #     data = json_graph.node_link_data(subgraph)
        #     with open(f'./results/log/2023y10m24d/subgraph_{i}.json', 'w') as output:
        #         json.dump(data, output, sort_keys=True, indent=4, separators=(',', ':'))
        # # Visualize the graph
        # for i, node in enumerate(nodes):
        #     plt.figure(figsize=(8, 6))
        #     subgraph = G.subgraph(node)
        #     plt.subplot(121)
        #     nx.draw(G, with_labels=True, node_color='lightblue',
        #             node_size=500, font_size=16, font_weight='bold')
        #     plt.title("Original Graph")

        #     plt.subplot(122)
        #     nx.draw(subgraph, with_labels=True, node_color='lightgreen', node_size=500)
        #     plt.title("Subgraph")
        #     print("Number of nodes:", G.number_of_nodes())
        #     print("Number of edges:", G.number_of_edges())

        #     plt.tight_layout()
        #     plt.title(f"Fidelity = {fidelity_plus_modif}, Sparsity = {len(node) / G.number_of_nodes()}")
        #     plt.savefig(f"./results/log/2023y10m24d/sub_{i}.png")

    def eval(self, input, real_val, lambdas: list = None):
        self.lambdas = lambdas
        self.model.eval()
        with ((torch.no_grad())):
            (logits, probs, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, min_distance,
             congestion, adp, importance, x_comp, sparsity, sampled_adj) = self.model(
                input)  # TODO: I added sampled_adj instead of _  --->  for self.graph =True
            
            output = logits.transpose(1, 3)
            fidelity_plus_modif = torch.abs(
                output - x_comp[0]).mean()  # TODO: check the code of fidility and avan va ansar
            fidelity_norm = torch.norm(output - x_comp[0])
            fidelity_plus_topk = torch.abs(output - x_comp[1]).mean()
            self.writer.add_scalar('Fidelity+/val', fidelity_plus_modif, self.iter)
            self.writer.add_scalar('Fidelity_Norm/val', fidelity_norm, self.iter)
            self.writer.add_scalar('Fidelity_topk/val', fidelity_plus_topk, self.iter)
            self.writer.add_scalar('Sparsity/val', sparsity, self.iter)
            # output = logits.transpose(1, 3)
            real = torch.unsqueeze(real_val, dim=1)
            if self.mi_loss:
                predict = output
            else:
                predict = self.scaler.inverse_transform(output)
            loss_reg = self.loss(predict, real, self.scaler, self.min_val, null_val=0.0,
                                 iner_coeff=1.0)  # TODO: change the name of loss to loss_reg
            # congestion_th = torch.from_numpy(self.congestion_th).to(args.device)
            # torch.Size([64, 1, 207, 12]) => real[:, :,:]
            congestion_gt = (real[:, :, :].mean(dim=-1).squeeze(0) < self.congestion_th.to(args.device)).squeeze(1).to(
                torch.long)

            try:
                congestion_loss = F.cross_entropy(congestion.transpose(1, 2), congestion_gt)
            except IndexError:
                congestion_loss = torch.tensor(0.0).to(real.device)

            if self.mi_loss:
                predict = inverse_transform(predict, self.scaler, self.min_val)

            mae = masked_mae(predict, real, 0.0).item()

            self.writer.add_scalar('MAE/val', mae, self.iter)

            mape = masked_mape(predict, real, 0.0).item()
            rmse = masked_rmse(predict, real, 0.0).item()

            ##################################################################

            if self.graph:
                graph_loss = torch.nn.MSELoss()(self.model.predefined_A +
                                                (torch.eye(self.model.predefined_A[1].shape[0]).to(
                                                    self.model.predefined_A.device).repeat(sampled_adj.shape[0], 1, 1)),
                                                sampled_adj)
                self.writer.add_scalar('graph_reconstruction/val', graph_loss.mean(), self.iter)

            # if args.CoVWeightingLoss:
            #     self.criterion.to_eval()
            #     losses_list = [loss_reg.mean(), prototype_pred_loss.mean(), connectivity_loss.mean(),
            #                    KL_Loss.mean(), congestion_loss.mean()]
            #
            #     if self.graph:
            #         losses_list.append(graph_loss.mean())
            #
            #     comb = self.criterion.forward(losses_list)

            comb = loss_reg.mean() + self.lambdas[0] * prototype_pred_loss.mean()
            + self.lambdas[1] * connectivity_loss.mean()
            + self.lambdas[2] * KL_Loss.mean()
            + self.lambdas[3] * congestion_loss.mean()
            if self.graph:
                comb = comb + self.lambdas[3] * graph_loss.mean()  # TODO: self.lambdas[3] ?? OR self.lambdas[4] ?

            ##################################################################

            self.writer.add_scalar('congestion_loss/val', congestion_loss.mean().detach().item(), self.iter)
            self.writer.add_scalar('KL_Loss/val', KL_Loss.mean().detach().item(), self.iter)
            self.writer.add_scalar('prototype_pred_loss/val', prototype_pred_loss.mean().detach().item(), self.iter)
            self.writer.add_scalar('connectivity_loss/val', connectivity_loss.mean().detach().item(), self.iter)
            self.writer.add_scalar('reg_loss/val', loss_reg.mean().detach().item(), self.iter)
            return comb.item(), mape, rmse, comb


# class Optim(object):

#     def _makeOptimizer(self):
#         if self.method == 'sgd':
#             self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
#         elif self.method == 'adagrad':
#             self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
#         elif self.method == 'adadelta':
#             self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
#         elif self.method == 'adam':
#             self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
#         else:
#             raise RuntimeError("Invalid optim method: " + self.method)

#     def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
#         self.params = params  # careful: params may be a generator
#         self.last_ppl = None
#         self.lr = lr
#         self.clip = clip
#         self.method = method
#         self.lr_decay = lr_decay
#         self.start_decay_at = start_decay_at
#         self.start_decay = False

#         self._makeOptimizer()

#     def step(self):
#         # Compute gradients norm.
#         grad_norm = 0
#         if self.clip is not None:
#             torch.nn.utils.clip_grad_norm_(self.params, self.clip)

#         # for param in self.params:
#         #     grad_norm += math.pow(param.grad.data.norm(), 2)
#         #
#         # grad_norm = math.sqrt(grad_norm)
#         # if grad_norm > 0:
#         #     shrinkage = self.max_grad_norm / grad_norm
#         # else:
#         #     shrinkage = 1.
#         #
#         # for param in self.params:
#         #     if shrinkage < 1:
#         #         param.grad.data.mul_(shrinkage)
#         self.optimizer.step()
#         return grad_norm

#     # decay learning rate if val perf does not improve or we hit the start_decay_at limit
#     def updateLearningRate(self, ppl, epoch):
#         if self.start_decay_at is not None and epoch >= self.start_decay_at:
#             self.start_decay = True
#         if self.last_ppl is not None and ppl > self.last_ppl:
#             self.start_decay = True

#         if self.start_decay:
#             self.lr = self.lr * self.lr_decay
#             print("Decaying learning rate to %g" % self.lr)
#         # only decay for one epoch
#         self.start_decay = False

#         self.last_ppl = ppl

#         self._makeOptimizer()


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0', help='')
# parser.add_argument('--data', type=str, default='datasets/METRLA', help='data path')
parser.add_argument('--data', type=str, default='datasets/PEMS04', help='data path')
parser.add_argument('--hparam_file', type=str, default=None, help='hparam path')

parser.add_argument('--adj_data', type=str,
                    default='./datasets/PEMS04/PEMS04.csv',
                    help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,
                    help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False, help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True, help='whether to do curriculum learning')

parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--desc', type=str, default=None, help='Describe run')
parser.add_argument('--num_nodes', type=int, default=307, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')

parser.add_argument('--conv_channels', type=int, default=32, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=32, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=64, help='skip channels')
parser.add_argument('--end_channels', type=int, default=128, help='end channels')

parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')

parser.add_argument('--layers', type=int, default=3, help='number of layers')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_per_epoch', type=int, default=None, help='learning rate decay rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', nargs='?', type=int, default=None, const=None, help='clip')
parser.add_argument('--step_size1', type=int, default=2500, help='step_size')
parser.add_argument('--step_size2', type=int, default=100, help='step_size')

parser.add_argument('--epochs', type=int, default=300, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--test_every', type=int, default=5, help='')
parser.add_argument('--seed', type=int, default=101, help='random seed')
parser.add_argument('--save', type=str, default='./save/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')

parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')

parser.add_argument('--runs', type=int, default=10, help='number of runs')
parser.add_argument("--inference", type=str, default=None, help='whether to do inference')
parser.add_argument("--resume", type=str, default=None, help='whether to do inference')
parser.add_argument("--optuna", type=str_to_bool, default=False, help='hyperparameter tuning')
parser.add_argument("--annealing", type=str_to_bool, default=False, help='annealling method instead of optuna')

parser.add_argument("--num_classes", type=int, default=2, help='num_classes')

parser.add_argument("--proto_percnetile", type=float, default=0.1, help='proto_percnetile')
parser.add_argument("--COUNT_THRESHOLD", type=int, default=1, help='proto_percnetile')

parser.add_argument("--proj", type=str_to_bool, default=False, help='Aplly proejction OR NOT')
parser.add_argument("--xai", type=str_to_bool, default=True, help='XAI on OR NOT')
parser.add_argument("--proj_epochs", type=int, default=1, help='proj_epochs number')
parser.add_argument("--share_merge", type=str_to_bool, default=False, help='Aplly merging OR NOT')
parser.add_argument("--merge_p", type=float, default=0.3, help='merge_p')
parser.add_argument("--gsub", type=str_to_bool, default=True, help='Aplly gsub OR NOT')
parser.add_argument("--proto", type=str_to_bool, default=True, help='gsub_epochs number')
parser.add_argument("--addaptive_coef", type=str_to_bool, default=False, help='addaptive_coef Branch')
parser.add_argument("--mi_loss", type=str_to_bool, default=False, help='Ali Loss IB')
parser.add_argument("--loss_mse", type=int, help='MSE Loss IB')
parser.add_argument("--graph_reg", type=str_to_bool, default=False, help='Graph Regularization')
parser.add_argument("--mean_pool", type=str_to_bool, default=False, help='Mean pool')
parser.add_argument("--CoVWeightingLoss", type=str_to_bool, default=False, help='CoVWeighting for losses'
                                                                               'coefficients')
parser.add_argument('--mean_sort', type=str, help='CoVWeightingLoss: full or decay', default='full')
parser.add_argument('--CoVWeightingLoss_custom', type=str_to_bool, help='exclude reg_loss, include others', default=False)
parser.add_argument('--set_alpha_init', type=str_to_bool, help='set alphas value from lambdas hyperparam'
                                                               ' file created from optuna', default=False)
parser.add_argument('--mean_decay_param', type=float, help='CoVWeightingLoss: '
                                                           'What decay to use with mean decay', default=1.0)


parser.add_argument('--PCGrad', type=str_to_bool, help='PCGrad method', default=False)

parser.add_argument('--UncertaintyMethod', type=str_to_bool, help='UncertaintyMethod for losses'
                                                                  ' coefficients', default=False) #TODO

parser.add_argument("--scheduler", type=str, default=None, help='optimizer scheduler: "ReduceLROnPlateau", '
                                                                '"exponential", "cosine"')
parser.add_argument("--optimizer", type=str, default='Adam', help='optimizer: "adam", "adamw')
parser.add_argument("--swa", type=str_to_bool, default=False, help='Stochastic Weight Averaging ---> '
                                                                   'https://pytorch.org/docs/stable/optim.html')
parser.add_argument("--ema", type=str_to_bool, default=False, help='Exponential Moving Average --->'
                                                                   'https://pytorch.org/docs/stable/optim.html')#TODO
parser.add_argument("--visualize", type=str_to_bool, default=False, help='Visualize the graph')
parser.add_argument("--F_S", type=str, default="test_loader", help='optimizer scheduler: "test_loader", '
                                                                '"val_loader", "train_loader"')
# parser.add_argument("--two_branch", type=str_to_bool, default=True, help='Ali Loss IB')

args = parser.parse_args()
torch.set_num_threads(3)

if args.graph_reg:
    parser.add_argument('--num_losses', type=int, help='CoVWeightingLoss: number of losses', default=6)
else:
    parser.add_argument('--num_losses', type=int, help='CoVWeightingLoss: number of losses', default=5)
args = parser.parse_args()

DEFAULT_NUM_LOSSES = args.num_losses

if args.CoVWeightingLoss_custom:
    args.num_losses = args.num_losses - 1



# self.writer = SummaryWriter()

def Beta_Function(x, alpha, beta):
    """Beta function"""
    from scipy.special import gamma
    return gamma(alpha + beta) / gamma(alpha) / gamma(beta) * x ** (alpha - 1) * (1 - x) ** (beta - 1)


def build_beta_list(max_epoch, beta1=1, beta2=1, beta3=1, beta4=1):
    beta_init = 0
    init_length = int(max_epoch / 4)
    anneal_length = int(max_epoch / 4)
    beta_inter = Beta_Function(np.linspace(0, 1, anneal_length), 1, 4)
    beta1_inter = beta_inter / 4 * (beta_init - beta1) + beta1
    beta1_list = np.concatenate([np.ones(init_length) * beta_init, beta1_inter,
                                 np.ones(max_epoch - init_length - anneal_length + 1) * beta1])

    beta_init = 0
    init_length = int(max_epoch / 4)
    anneal_length = int(max_epoch / 4)
    beta_inter = Beta_Function(np.linspace(0, 1, anneal_length), 1, 4)
    beta2_inter = beta_inter / 4 * (beta_init - beta2) + beta2
    beta2_list = np.concatenate([np.ones(init_length) * beta_init, beta2_inter,
                                 np.ones(max_epoch - init_length - anneal_length + 1)
                                 * beta2])

    beta_init = 0
    init_length = int(max_epoch / 4)
    anneal_length = int(max_epoch / 4)
    beta_inter = Beta_Function(np.linspace(0, 1, anneal_length), 1, 4)
    beta3_inter = beta_inter / 4 * (beta_init - beta3) + beta3
    beta3_list = np.concatenate([np.ones(init_length) * beta_init, beta3_inter,
                                 np.ones(max_epoch - init_length - anneal_length + 1)
                                 * beta3])

    beta_init = 0
    init_length = int(max_epoch / 4)
    anneal_length = int(max_epoch / 4)
    beta_inter = Beta_Function(np.linspace(0, 1, anneal_length), 1, 4)
    beta4_inter = beta_inter / 4 * (beta_init - beta4) + beta4
    beta4_list = np.concatenate([np.ones(init_length) * beta_init, beta4_inter,
                                 np.ones(max_epoch - init_length - anneal_length + 1)
                                 * beta4])
    return beta1_list, beta2_list, beta3_list, beta4_list


def get_last_commit_hash():
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting last commit hash: {e}")
        return None


def get_module_file_path(module):
    """Gets the file path of a Python module.

    Args:
      module: The module object.

    Returns:
      The path to the module's file as a string.
    """

    if hasattr(module, '__file__'):
        return os.path.abspath(module.__file__)
    else:
        raise ValueError(f"Module {module.__name__} has no __file__ attribute.")


def write_metadata(model, args, path, hparam):
    metadata = {
        'model_settings': {
            'gcn_true': args.gcn_true,
            'buildA_true': args.buildA_true,
            'gcn_depth': args.gcn_depth,
            'num_nodes': args.num_nodes,
            'device': args.device,
            'predefined_A': getattr(args, "predefined_A", None),
            "static_feat": getattr(args, "load_static_feature", None),
            'dropout': getattr(args, "dropout", 0.3),
            'subgraph_size': getattr(args, "subgraph_size", 20),
            'node_dim': getattr(args, "node_dim", 40),
            'dilation_exponential': getattr(args, "dilation_exponential", 1),
            'conv_channels': getattr(args, "conv_channels", 32),
            'residual_channels': getattr(args, "residual_channels", 32),
            'skip_channels': getattr(args, "skip_channels", 64),
            'end_channels': getattr(args, "end_channels", 128),
            'seq_length': getattr(args, "seq_in_len", 12),
            'in_dim': getattr(args, "in_dim", 2),
            'out_dim': getattr(args, "seq_out_len", 12),
            'layers': getattr(args, "layers", 3),
            'propalpha': getattr(args, "propalpha", 0.05),
            'tanhalpha': getattr(args, "tanhalpha", 3),
            'addaptive_coef': getattr(args, "addaptive_coef", False),
            'mi_loss': getattr(args, "mi_loss", False),
            'loss_mse': getattr(args, "loss_mse", False),
            'layer_norm_affline': getattr(args, "layer_norm_affline", True),
            "xai": {"xai": args.xai, "gsub": args.gsub, "proto": args.proto}
        },
        "model_dict": [(item[0], str(item[1])) for item in inspect.getmembers(model) if
                       not hasattr(item[1], "__call__") and not isinstance(item[1], torch.Tensor) and not str(
                           item[0]).startswith("__") and not str(item[0]).startswith("_")],
        'args': vars(args),
        "dir": path,
        "commit_id": get_last_commit_hash(),
        "hparams": hparam
    }
    with open(f'{path}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    with open(f'{path}/model_revision.py', 'w') as f:
        f.write(inspect.getsource(model.__class__))


def engine_factory(epoch, model, scaler, min_val, device, congestion, writer):
    engine = Trainer(
        epoch,
        model,
        args.learning_rate,
        args.weight_decay,
        args.clip,
        args.step_size1,
        args.seq_out_len,
        scaler,
        min_val,
        device,
        args.cl,
        congestion_th=congestion,
        writer=writer,
        mi_loss=args.mi_loss,
        loss_mse=args.loss_mse,
        graph_reg=args.graph_reg
    )
    return engine


def main_pems04(trial):
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.seed)
    # load data
    # 1/0
    lamda_1, lamda_2, lamda_3, lamda_4 = 0.0, 0.0, 0.0, 0.0

    current_datetime = datetime.datetime.now()
    datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs", f"{datetime_string}_trial_no_{trial.number}")
    epoch_t = args.epochs
    if args.optuna:
        print("Start Trial", trial.number, "with parameters", trial.params, flush=True)
        # lamda_1 = trial.suggest_float('lamda_1', 1e-5, 1e-1)
        # lamda_2 = trial.suggest_float('lamda_2', 1, 10)
        # lamda_3 = trial.suggest_float('lamda_3', 1e-5, 1e-3)
        # lamda_4 = trial.suggest_float('lamda_4', 1e-5, 1e-3)

        # # GPU 0
        # lamda_1 = trial.suggest_float('lamda_1', 1e-3, 1e-1)
        # lamda_2 = trial.suggest_float('lamda_2', 5, 10)
        # lamda_3 = trial.suggest_float('lamda_3', 1e-4, 1e-3)
        # lamda_4 = trial.suggest_float('lamda_4', 1e-4, 1e-3)
        # GPU 1
        # 0.000485432334023971, "lamda_2": 1.2429058276346123, "lamda_3": 7.011798342265815e-05, "lamda_4": 9.758598058413018e-05
        lamda_1 = trial.suggest_float('lamda_1', 0.000485432334023971 / 4, 4*0.000485432334023971)
        lamda_2 = trial.suggest_float('lamda_2', 3.7001070264887375/8, 3.7001070264887375 * 8)
        lamda_3 = trial.suggest_float('lamda_3', 0.00026950420419019616/8, 0.00026950420419019616 * 8)
        lamda_4 = trial.suggest_float('lamda_4', 0.00015120958136040239, 0.00015120958136040239 * 4)
        # num_prototypes_per_class = trial.suggest_categorical('num_prototypes_per_class', [2, 4, 8, 16, 32])  # TODO
        args.scheduler = trial.suggest_categorical('scheduler', ['ReduceLROnPlateau', 'cosine'])
        args.addaptive_coef = trial.suggest_categorical('addaptive_coef', [True, False])
        args.optim = trial.suggest_categorical('optimizer', ['Adam', 'Adadelta'])
        args.loss_mse = trial.suggest_categorical('loss_mse', [0, 1, 2, None])
        args.CoVWeightingLoss_custom = trial.suggest_categorical('CoVWeightingLoss_custom', [True, False])
        if args.CoVWeightingLoss_custom:
            args.CoVWeightingLoss = True
            args.num_losses = DEFAULT_NUM_LOSSES - 1
        else:
            args.num_losses = DEFAULT_NUM_LOSSES
            args.CoVWeightingLoss = False
        args.mean_pool = trial.suggest_categorical('mean_pool', [True, False])
        # args.mi_loss = trial.suggest_categorical('mi_loss', [True, False])
        args.layers = trial.suggest_categorical('layers', [1, 3])
        args.conv_channels = trial.suggest_categorical('conv_channels', [64, 32])
        args.residual_channels = trial.suggest_categorical('residual_channels', [16, 32])
        args.gcn_depth = trial.suggest_categorical('gcn_depth', [4, 2])
        ####################################################
        epoch_t = args.epochs
        if trial.number < 150:
            epoch_t = 30
            MAX_ITER = 80
        elif trial.number < 400:
            epoch_t = 50
            MAX_ITER = 100
        else:
            epoch_t = 100
            MAX_ITER = None
        # if trial.number < 10:
        #     epoch_t = 1
        #     MAX_ITER = 2
        # elif trial.number < 80:
        #     epoch_t = 20
        #     MAX_ITER = 100
        # else:
        #     epoch_t = 100
        #     MAX_ITER = None
        # args.cl = False
        # args.device = trial.suggest_categorical('device', ['cuda:0', 'cuda:1'])
        ###################################################

        with open(f'./hyperparams/hyperparams_trial_no_{trial.number}_{datetime_string}.json',
                  'w') as f:
            json.dump({'lamda_1': lamda_1, 'lamda_2': lamda_2, 'lamda_3': lamda_3, 'lamda_4': lamda_4}, f)




    elif args.annealing:
        beta1_list, beta2_list, beta3_list, beta4_list = build_beta_list(args.epochs, 1, 1, 1, 1)


    else:
        print(args, flush=True)
        with open(args.hparam_file or 'hyperparams/hyperparams_trial_no_3_2024-07-26_15-05-54.json', 'r') as f:
            file = json.load(f)
        lamda_1 = file['lamda_1']
        lamda_2 = file['lamda_2']
        lamda_3 = file['lamda_3']
        lamda_4 = file['lamda_4']
        # num_prototypes_per_class = file['num_prototypes_per_class'] #TODO
        num_prototypes_per_class = 7
    if not args.xai:
        lamda_1, lamda_2, lamda_3, lamda_4 = 0.0, 0.0, 0.0, 0.0
    else:
        if not args.proto:
            lamda_1 = 0.0
            lamda_4 = 0.0  # TODO: check
        if not args.gsub:
            lamda_2 = 0.0
            lamda_3 = 0.0
        if args.CoVWeightingLoss or args.PCGrad:
            lamda_1 = 1.0
            lamda_2 = 1.0
            lamda_3 = 1.0
            lamda_4 = 1.0

    device = torch.device(args.device)
    train_index = None
    if args.batch_per_epoch:
        train_index = args.batch_per_epoch * args.batch_size
    (_, train_loader, _, _, val_loader, val_target_tensor, _, test_loader,
     test_target_tensor, _mean, _std, congestion, min_val) = load_graphdata_channel1(args.data + "/PEMS04.npz" if not args.data.endswith("npz") else args.data, 1,
                                                                                     0, 0, device, args.batch_size)

    _mean, _std = torch.tensor(_mean).to(device), torch.tensor(_std).to(device)

    congestion_th = torch.from_numpy(congestion).to(device)


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
    # if args.adj_data:
    matrix = predefined_A = get_adjacency_matrix2(args.adj_data, args.num_nodes,
                                         id_filename=None)  # TODO: (idx) --> for sub_grapf augmentaion
    predefined_A = torch.tensor(predefined_A) - torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)
    # Format the datetime as a string

    writer = SummaryWriter(log_dir=log_dir)

    print("Lamdas are", lamda_1, lamda_2, lamda_3, lamda_4)
    print(log_dir)

    # if args.load_static_feature:
    #     static_feat = load_node_feature('data/sensor_graph/location.csv')
    # else:
    #     static_feat = None

    # GIB model
    # model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
    #               device, predefined_A=predefined_A,
    #               dropout=args.dropout, subgraph_size=args.subgraph_size,
    #               node_dim=args.node_dim,
    #               dilation_exponential=args.dilation_exponential,
    #               conv_channels=args.conv_channels, residual_channels=args.residual_channels,
    #               skip_channels=args.skip_channels, end_channels= args.end_channels,
    #               seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
    #               layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    # Our model
    model = ProtoGtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                       device, predefined_A=predefined_A,
                       dropout=args.dropout, subgraph_size=args.subgraph_size,
                       node_dim=args.node_dim,
                       dilation_exponential=args.dilation_exponential,
                       conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                       skip_channels=args.skip_channels, end_channels=args.end_channels,
                       seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                       layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True,
                       is_xai={"xai": args.xai, "gsub": args.gsub, "proto": args.proto}, addaptive_coef=args.addaptive_coef,
                       mean_pool=args.mean_pool, graph_reg=args.graph_reg
                       )

    if args.resume:
        model_path = args.resume
        model.load_state_dict(torch.load(model_path))

    write_metadata(model, args, log_dir, hparam=[lamda_1, lamda_2, lamda_3, lamda_4])
    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)
    scalar = StandardScaler(mean=_mean, std=_std)
    engine = engine_factory(args.epochs, model, scalar, min_val, device, congestion, writer)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    minl = 1e5
    # from collections import Slice
    train_index = None
    if args.batch_per_epoch:
        train_index = args.batch_per_epoch * args.batch_size

    # dataset_org = load_dataset(args.data, 1, 1, 1, train_index)['train_loader']
    # dataset_org = dataloader['train_loader']
    if not args.inference:

        for i in range(0, epoch_t + 1):

            if args.proj:

                if i >= args.proj_epochs and i % args.proj_epochs == 0:
                    model.eval()
                    # if is_congested(y):

                    for proto_index in range(model.prototype_vectors.shape[0]):
                        count = 0
                        best_similarity = 0
                        label = model.prototype_class_identity[0].max(0)[1]  # TODO
                        # for j in range(proto_index*10, len(dataset_org)): 

                        for iter, (data, y) in enumerate(train_loader):
                            ############################

                            ############################
                            y = torch.Tensor(y).to(device)
                            y = y.transpose(1, 3)[:, 0, :, :]

                            # congestion_th = torch.from_numpy(engine.congestion_th).to(y.device)
                            if iter >= proto_index * 10:
                                # data = dataloader['train_loader']
                                real = torch.unsqueeze(y, dim=1)
                                congestion_gt = (real[:, :, :].mean(dim=-1).squeeze(0) <
                                                 congestion_th).squeeze(1).to(torch.long)
                                if congestion_gt == label:
                                    count += 1
                                    coalition, similarity, prot = mcts(data, model,
                                                                       model.prototype_vectors[proto_index], index=id)
                                    if similarity > best_similarity:
                                        best_similarity = similarity
                                        proj_prot = prot
                                if count >= args.COUNT_THRESHOLD:
                                    model.prototype_vectors.data[proto_index] = proj_prot
                                    print(f'Projection of prototype {proto_index} completed')
                                    break
            # prototype merge
            if args.share_merge:
                if model.prototype_vectors.shape[0] > round(
                        args.num_classes * num_prototypes_per_class * (1 - args.merge_p)):
                    join_info = join_prototypes_by_activations(model, args.proto_percnetile, train_loader,
                                                               engine.optimizer)

            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            # if not args.batch_per_epoch:
            #     dataloader['train_loader'].shuffle()

            for iter, (x, y) in enumerate(train_loader):
                if args.optuna and (MAX_ITER is not None and iter > MAX_ITER):
                    break

                trainx = x.squeeze(2).transpose(1, 2).unsqueeze(-1)
                trainx = trainx.transpose(1, 3)
                trainy = y.squeeze(2).transpose(1, 2).unsqueeze(-1)

                # breakpoint() #  torch.Size([64, 307, 12])
                trainy = trainy.transpose(1, 3)
                if iter % args.step_size2 == 0:
                    perm = np.random.permutation(range(args.num_nodes))
                num_sub = int(args.num_nodes / args.num_split)
                count = 0
                best_similarity = 0
                for j in range(args.num_split):  # TODO: (idx) --> for sub_grapf augmentaion
                    if j != args.num_split - 1:
                        id = perm[j * num_sub:(j + 1) * num_sub]
                    else:
                        id = perm[j * num_sub:]
                    id = torch.tensor(id).to(device)
                    tx = trainx[:, :, id, :]
                    ty = trainy[:, :, id, :]
                    # if iter == 1:
                    #     breakpoint()
                    if args.annealing:
                        lamda_1, lamda_2, lamda_3, lamda_4 = beta1_list[i], beta2_list[i], beta3_list[i], beta4_list[i]
                    metrics = engine.train(tx, ty[:, 0, :, :], id, lambdas=[lamda_1, lamda_2, lamda_3, lamda_4],
                                           train_loader_len=len(train_loader))
                    data = Data(tx, dense_to_sparse(metrics[3])[0])
                    train_loss.append(metrics[0])
                    train_mape.append(metrics[1])
                    train_rmse.append(metrics[2])
                    # Should be tested

                if iter % args.print_every == 0:
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

            t2 = time.time()
            train_time.append(t2 - t1)
            # validation
            valid_loss = []
            valid_mape = []
            valid_rmse = []

            s1 = time.time()
            for iter, (x, y) in enumerate(val_loader):
                testx = x.squeeze(2).transpose(1, 2).unsqueeze(-1)
                testx = testx.transpose(1, 3)
                testy = y.squeeze(2).transpose(1, 2).unsqueeze(-1)
                testy = testy.transpose(1, 3)


                if args.annealing:
                    lamda_1, lamda_2, lamda_3, lamda_4 = beta1_list[i], beta2_list[i], beta3_list[i], beta4_list[i]
                metrics = engine.eval(testx, testy[:, 0, :, :], lambdas=[lamda_1, lamda_2, lamda_3, lamda_4])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])

            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i, (s2 - s1)))
            val_time.append(s2 - s1)

            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            # if mtrain_loss == np.nan or mtrain_mape == np.nan or mtrain_rmse == np.nan:
            #     print("#############")

            mvalid_loss = np.mean(valid_loss)

            if args.swa:
                if i > engine.swa_start:
                    engine.swa_model.update_parameters(engine.model)
                    engine.swa_scheduler.step()
                else:
                    engine.scheduler.step()

            if args.scheduler and not args.swa:

                if args.scheduler == "ReduceLROnPlateau":
                    engine.scheduler.step(mvalid_loss)
                else:
                    engine.scheduler.step()



            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            # self.writer.add_scalar('Loss/test', metrics[0], iter)
            # self.writer.add_scalar('MAPE/test', metrics[1], iter)
            # self.writer.add_scalar('RMSE/test', metrics[2], iter)
            writer.add_scalar('Loss/train', mtrain_loss, i)
            writer.add_scalar('MAPE/train', mtrain_mape, i)
            writer.add_scalar('RMSE/train', mtrain_rmse, i)
            writer.add_scalar('Loss/valid', mvalid_loss, i)
            writer.add_scalar('MAPE/valid', mvalid_mape, i)
            writer.add_scalar('RMSE/valid', mvalid_rmse, i)
            his_loss.append(mvalid_loss)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(
                log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
                flush=True)

            ###### Test the model on the test data
            if i % args.test_every == 0:

                reg_loss_list = 0.0
                loss_list = 0.0
                congestion_loss_list = 0.0
                KL_Loss_list = 0.0
                prototype_pred_loss_list = 0.0
                connectivity_loss_list = 0.0
                masked_mae_list = 0.0
                masked_mape_list = 0.0
                masked_rmse_list = 0.0

                model.eval()
                with ((torch.no_grad())):
                    for iter, (x, y) in enumerate(test_loader):

                        testx = x.squeeze(2).transpose(1, 2).unsqueeze(-1)
                        testx = testx.transpose(1, 3)

                        if args.swa:
                            if i > engine.swa_start:
                                (preds, probs, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, min_distance,
                                 congestion, adp, importance, x_comp, sparsity, _) = engine.swa_model(testx)
                            else:
                                (preds, probs, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, min_distance,
                                 congestion, adp, importance, x_comp, sparsity, _) = engine.model(testx)

                        else:
                            (preds, probs, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, min_distance,
                            congestion, adp, importance, x_comp, sparsity, _ )= engine.model(testx)

                        preds = preds.transpose(1, 3)
                        # if args.mi_loss:
                        #     preds = inverse_transform(preds, scalar, min_val)
                        testy = y.squeeze(2).transpose(1, 2).unsqueeze(-1)
                        # One bug is here, the congestion
                        testy = testy.transpose(1, 3)


                        # congestion_th = torch.from_numpy(congestion_th).to(
                        #     testy.device)

                        congestion_gt = (
                                testy[:, 0, :, :].mean(dim=-1).squeeze(0).transpose(1, -1) < congestion_th).squeeze(
                            1).to(
                            torch.long)

                        try:
                            congestion_loss = F.cross_entropy(congestion.transpose(1, 2),
                                                              congestion_gt)  # TODO: congestion in every condition should be done
                        except Exception as e:
                            congestion_loss = torch.tensor(0.0, device=device)
                        if args.mi_loss:
                            reg_loss = ib_masked_mae(preds, testy, scalar, min_val, null_val=0.0, iner_coeff=1.0).mean()
                            preds = inverse_transform(preds, scalar, min_val)

                        else:
                            preds = scalar.inverse_transform(preds)
                            reg_loss = masked_mae(preds, testy, 0.0).mean()

                        loss = reg_loss
                        + lamda_1 * prototype_pred_loss.mean()
                        + lamda_2 * connectivity_loss.mean()
                        + lamda_3 * KL_Loss.mean()
                        + lamda_4 * congestion_loss.mean()



                        reg_loss_list += reg_loss
                        loss_list += loss
                        congestion_loss_list += congestion_loss.mean()
                        KL_Loss_list += KL_Loss.mean()
                        prototype_pred_loss_list += prototype_pred_loss.mean()
                        connectivity_loss_list += connectivity_loss.mean()

                        mae_test = masked_mae(preds, testy, 0.0)
                        mape_test = masked_mape(preds, testy, 0.0)
                        rmse_test = masked_rmse(preds, testy, 0.0)
                        masked_mae_list += mae_test.mean()
                        masked_mape_list += mape_test.mean()
                        masked_rmse_list += rmse_test.mean()


                    # try:
                    num_iter = (iter + 1)*1.0
                    writer.add_scalar('reg_loss/test_temp', reg_loss_list.item()/num_iter, i)
                    # except Exception as e:
                    #     print("Exception happened in reg_loss/test_temp", e)
                    #     writer.add_scalar('reg_loss/test_temp', reg_loss, iter))

                    # try:
                    writer.add_scalar('Loss/test_temp', loss_list.item()/num_iter, i)
                    writer.add_scalar('congestion_loss/test_temp', congestion_loss_list.detach().item()/num_iter, i)
                    writer.add_scalar('KL_Loss/test_temp', KL_Loss_list.detach().item()/num_iter, i)
                    writer.add_scalar('prototype_pred_loss/test_temp',
                                      prototype_pred_loss_list.detach().item()/num_iter, i)
                    writer.add_scalar('connectivity_loss/test_temp', connectivity_loss_list.detach().item()/num_iter,
                                      i)

                    writer.add_scalar('MAE/test_temp', masked_mae_list.detach().item()/num_iter, i)
                    writer.add_scalar('MAPE/test_temp', masked_mape_list.detach().item()/num_iter, i)
                    writer.add_scalar('RMSE/test_temp', masked_rmse_list.detach().item()/num_iter, i)
                    # except Exception as e:
                    #     writer.add_scalar('loss/test_temp', loss, iter)

            if not args.optuna and mvalid_loss < minl:
                torch.save(engine.model.state_dict(),
                           args.save + "exp" + str(args.expid) + "_" + str(trial.number) + ".pth")
                minl = mvalid_loss
                if args.annealing:
                    print("The Hyperparameters are", beta1_list[i], beta2_list[i], beta3_list[i], beta4_list[i])

            ###############################################
            if args.optuna:
                trial.report(mvalid_rmse, i)
                if trial.should_prune():
                        raise optuna.TrialPruned()
            ##############################################
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        bestid = np.argmin(his_loss)
        # model_path = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
        print("Training finished")
        print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
        if args.annealing:
            print("The Hyperparameters are", beta1_list[bestid], beta2_list[bestid], beta3_list[bestid],
                  beta4_list[bestid])

        if args.swa:
            try:
                print("$$$$$ updating Bach Normalization in 'optim.swa_utils.update_bn(train_loader, engine.swa_model)' $$$$")
                optim.swa_utils.update_bn(train_loader, engine.swa_model)
            except:
                print("$$$$$ Failed to update BN in 'optim.swa_utils.update_bn(train_loader, engine.swa_model)' $$$$")
                pass
    else:
        i = args.epochs
        model_path = args.inference

        if args.swa:
            engine.swa_model.load_state_dict(torch.load(model_path, map_location=args.device))  # TODO
        else:
            engine.model.load_state_dict(torch.load(model_path, map_location=args.device))  # TODO
        mtrain_loss = []
        print("Start testing best model", model_path)
        writer = SummaryWriter(log_dir=os.path.join("inference_mode_") + log_dir) #TODO: make it hierarchical. Means,...
        #TODO: ... put this wrire to related run sub folder
    # engine.model.load_state_dict(torch.load(model_path))
    # #valid data
    # test data

    #######################################################################
    # FIDILITY vs SPARSITY:


    if args.F_S:
        def inference_model(input, k_sparsity):
            if args.swa:
                (preds, probs, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, min_distance,
                     congestion, adp, importance, x_comp, sparsity, _) = engine.swa_model(input, k_sparsity=k_sparsity)
            else:
                (preds, probs, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, min_distance,
                 congestion, adp, importance, x_comp, sparsity, _) = engine.model(input, k_sparsity=k_sparsity)

            return x_comp

        def inference_run(LOADER, k_sparsity = 0.4):
            fidelity_minus_local = 0.0
            fidelity_minus_norm_local = 0.0
            fidelity_plus_local = 0.0
            fidelity_plus_norm_local = 0.0
            for iter, (x, y) in enumerate(LOADER):
                testx = x.squeeze(2).transpose(1, 2).unsqueeze(-1)
                testx = testx.transpose(1, 3)
                # testy = y.squeeze(2).transpose(1, 2).unsqueeze(-1)
                # testy = testy.transpose(1, 3)

                x_comp = inference_model(testx, k_sparsity)

                preds = x_comp[2]

                fidelity_minus_local += torch.abs(preds - x_comp[0]).mean()
                fidelity_minus_norm_local += (torch.norm(preds - x_comp[1], p=2, dim=-1) / math.sqrt(args.seq_out_len)).mean()
                fidelity_plus_local += torch.abs(preds - x_comp[1]).mean()
                fidelity_plus_norm_local += (torch.norm(preds - x_comp[1], p=2, dim=-1) / math.sqrt(args.seq_out_len)).mean()

            iter += 1
            return (fidelity_minus_local.detach().to("cpu").numpy() / iter, fidelity_minus_norm_local.detach().to("cpu").numpy() / iter,
                    fidelity_plus_local.detach().to("cpu").numpy() / iter, fidelity_plus_norm_local.detach().to("cpu").numpy() / iter)
        engine.model.eval()
        with torch.no_grad():
            if args.F_S in ["test_loader" , "val_loader" , "train_loader"]:
                if args.F_S == "test_loader":
                    LOADER = test_loader
                if args.F_S == "val_loader":
                    LOADER = val_loader
                if args.F_S == "train_loader":
                    LOADER = train_loader

                fidelity_minus_list = []
                fidelity_minus_norm_list = []
                fidelity_plus_list = []
                fidelity_plus_norm_list = []
                Sparsity_list = np.linspace(0.1, 0.9, num=20)

                for k_sparsity in tqdm(Sparsity_list):
                    (fidelity_minus, fidelity_minus_norm,
                    fidelity_plus, fidelity_plus_norm) = inference_run(LOADER, k_sparsity)

                    fidelity_minus_list.append(fidelity_minus)
                    fidelity_minus_norm_list.append(fidelity_minus_norm)
                    fidelity_plus_list.append(fidelity_plus)
                    fidelity_plus_norm_list.append(fidelity_plus_norm)

                for j, sparsity in enumerate(Sparsity_list):
                    writer.add_scalar(f'fidelity_minus_list/{args.F_S}', fidelity_minus_list[j], j)
                    writer.add_scalar(f'fidelity_minus_norm_list/{args.F_S}', fidelity_minus_norm_list[j], j)
                    writer.add_scalar(f'fidelity_plus_list/{args.F_S}', fidelity_plus_list[j], j)
                    writer.add_scalar(f'fidelity_plus_norm_list/{args.F_S}', fidelity_plus_norm_list[j], j)
                plt.plot(fidelity_minus_list, fidelity_plus_list, 'ro-', label="relative_fidelity")
                plt.xlabel("fidelity_minus")
                plt.ylabel("fidelity_plus")
                plt.savefig(f'./results/{args.F_S}_{uuid4()}_fidelity.png')
                # writer.add_scalar(f'fidelity_minus_list/{args.F_S}', np.array(fidelity_minus_list), Sparsity_list)
                # writer.add_scalar(f'fidelity_minus_norm_list/{args.F_S}', fidelity_minus_norm_list, Sparsity_list)
                # writer.add_scalar(f'fidelity_plus_list/{args.F_S}', fidelity_plus_list, Sparsity_list)
                # writer.add_scalar(f'fidelity_plus_norm_list/{args.F_S}', fidelity_plus_norm_list, Sparsity_list)


            elif args.F_S == "all":
                # for LOADER_name in ["test_loader", "val_loader", "train_loader"]:
                for LOADER_name in ["test_loader"]:
                    if LOADER_name == "test_loader":
                        LOADER = test_loader
                    if LOADER_name == "val_loader":
                        LOADER = val_loader
                    if LOADER_name == "train_loader":
                        LOADER = train_loader

                    fidelity_minus_list = []
                    fidelity_minus_norm_list = []
                    fidelity_plus_list = []
                    fidelity_plus_norm_list = []
                    Sparsity_list = np.linspace(0.01,1.0, num=50) #TODO: make it a parameter

                    for k_sparsity in tqdm(Sparsity_list):
                        #TODO: Multiprocessing
                        (fidelity_minus, fidelity_minus_norm,
                        fidelity_plus, fidelity_plus_norm) = inference_run(LOADER, k_sparsity)

                        fidelity_minus_list.append(fidelity_minus)
                        fidelity_minus_norm_list.append(fidelity_minus_norm)
                        fidelity_plus_list.append(fidelity_plus)
                        fidelity_plus_norm_list.append(fidelity_plus_norm)

                    for k, sparsity in enumerate(Sparsity_list):
                        writer.add_scalar(f'fidelity_minus_list/{LOADER_name}', fidelity_minus_list[k], k)
                        writer.add_scalar(f'fidelity_minus_norm_list/{LOADER_name}', fidelity_minus_norm_list[k], k)
                        writer.add_scalar(f'fidelity_plus_list/{LOADER_name}', fidelity_plus_list[k], k)
                        writer.add_scalar(f'fidelity_plus_norm_list/{LOADER_name}', fidelity_plus_norm_list[k], k)
                    plt.figure(figsize=(12, 6), dpi=300)
                    plt.rcParams.update({
                        "text.usetex": True,
                        "font.family": "serif",
                    })
                    plt.plot(np.array(fidelity_minus_list)*100, np.array(fidelity_plus_list)*100, 'ro-', label="relative_fidelity")
                    plt.xlabel("Fid_Neg", fontsize=14)
                    # Fix the y-label
                    plt.ylabel("Fid_Pos", fontsize=14)
                    plt.savefig(f'./results/relative_{LOADER_name}_{uuid4()}_fidelity.png')
                    np.save(f'./results/{LOADER_name}_fidelity_plus_list.npy', fidelity_plus_list)
                    np.save(f'./results/{LOADER_name}_sparsity.npy', Sparsity_list)
                    np.save(f'./results/{LOADER_name}_fidelity_minus_norm_list.npy', np.array(fidelity_minus_norm_list))
                    np.save(f'./results/{LOADER_name}_fidelity_plus_norm_list.npy', np.array(fidelity_plus_norm_list))
                    np.save(f'./results/{LOADER_name}_fidelity_minus_list.npy', np.array(fidelity_minus_list))


                # congestion_th = torch.from_numpy(congestion_th).to(
                #     testy.device)

                # congestion_gt = (
                #         testy[:, 0, :, :].mean(dim=-1).squeeze(0).transpose(1, -1) < congestion_th).squeeze(
                #     1).to(
                #     torch.long)
                #
                # try:
                #     congestion_loss = F.cross_entropy(congestion.transpose(1, 2),
                #                                       congestion_gt)  # TODO: congestion in every condition should be done
                # except Exception as e:
                #     congestion_loss = torch.tensor(0.0, device=device)



                # if args.mi_loss:
                #     reg_loss = ib_masked_mae(preds, testy, scalar, min_val, null_val=0.0, iner_coeff=1.0).mean()
                #     preds = inverse_transform(preds, scalar, min_val)
                #
                # else:
                #     preds = scalar.inverse_transform(preds)
                #     reg_loss = masked_mae(preds, testy, 0.0).mean()

                # loss = reg_loss
                # + lamda_1 * prototype_pred_loss.mean()
                # + lamda_2 * connectivity_loss.mean()
                # + lamda_3 * KL_Loss.mean()
                # + lamda_4 * congestion_loss.mean()




    ####################################################################

    outputs = []
    reg_losses_mean = 0.0
    losses_mean = 0.0
    realy = torch.Tensor(test_target_tensor).to(device)
    realy = realy.squeeze(2).transpose(1, 2).unsqueeze(-1).transpose(1, 3)[:, 0, :, :]
    model.eval()
    with torch.no_grad():
        for iter, (x, y) in enumerate(test_loader):
            testx = x.squeeze(2).transpose(1, 2).unsqueeze(-1)
            testx = testx.transpose(1, 3)

            if args.swa:
                (preds, probs, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, min_distance,
                 congestion, adp, importance, x_comp, sparsity, _) = engine.swa_model(testx)
            else:
                (preds, probs, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, min_distance,
                 congestion, adp, importance, x_comp, sparsity, _) = engine.model(testx)

            preds = preds.transpose(1, 3)
            testy = y.squeeze(2).transpose(1, 2).unsqueeze(-1)
            # One bug is here, the congestion
            testy = testy.transpose(1, 3)
            # congestion_th = torch.from_numpy(congestion_th).to(testy.device)
            congestion_gt = (testy[:, 0, :, :].mean(dim=-1).squeeze(0).transpose(1, -1) < congestion_th).squeeze(1).to(
                torch.long)
            try:
                congestion_loss = F.cross_entropy(congestion.transpose(1, 2), congestion_gt)
            except Exception as e:
                congestion_loss = torch.tensor(0.0, device=device)
            if args.visualize:
                Trainer.visualize(
                    testx,
                    adp,
                    probs, 
                    testy=testy,
                    preds=scalar.inverse_transform(preds), 
                    fidelity_plus_modif=torch.abs(testy - x_comp[0]).mean(), 
                    congestion=congestion, 
                    matrix=matrix,
                    iter_num=iter,
                    importance=importance,
                )
            if args.mi_loss:
                reg_loss = ib_masked_mae(preds, testy, scalar, min_val, null_val=0.0, iner_coeff=1.0).mean()
                preds = inverse_transform(preds, scalar, min_val)
            else:
                preds = scalar.inverse_transform(preds)
                reg_loss = masked_mae(preds, testy, 0.0).mean()
            try:
                writer.add_scalar('reg_loss/test', reg_loss.item(), i if not args.inference else iter)
            except Exception as e:
                print("Exception happened in reg_loss/test", e)
                writer.add_scalar('reg_loss/test', reg_loss, iter)
            loss = reg_loss
            + lamda_1 * prototype_pred_loss.mean()
            + lamda_2 * connectivity_loss.mean()
            + lamda_3 * KL_Loss.mean()
            + lamda_4 * congestion_loss.mean()

            try:
                writer.add_scalar('Loss/test', loss.item(), i)
            except Exception as e:
                print("Exception happened in Loss/test", e)
                writer.add_scalar('Loss/test', loss, iter)
            outputs.append(preds.squeeze())

            reg_losses_mean += reg_loss
            losses_mean += loss

        print("Overall results, reg_loss: {:.4f}, Loss: {:.4f}".format(reg_losses_mean / len(test_loader),
                                                                       losses_mean / len(test_loader)))


    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]  # TODO (important
    mae = []
    mape = []
    rmse = []
    for i in range(args.seq_out_len):
        pred = yhat[:, :, i]
        real = realy[:, :, i]
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
        writer.add_scalar('MAE/test', metrics[0], i)
        writer.add_scalar('MAPE/test', metrics[1], i)
        writer.add_scalar('RMSE/test', metrics[2], i)
    writer.add_hparams(
        {"lamda_1": lamda_1, "lamda_2": lamda_2, "lamda_3": lamda_3, "lamda_4": lamda_4,
        #  "num_prototypes_per_class": num_prototypes_per_class,
         "num_classes": args.num_classes,
         "batch_size": args.batch_size, "epochs": args.epochs, "gcn_depth": args.gcn_depth, "layers": args.layers,
         "addaptive_coef": args.addaptive_coef, "mi_loss": args.mi_loss, "loss_mse": args.loss_mse, "xai": args.xai,
         "gsub": args.gsub, "proto": args.proto, "graph_reg": args.graph_reg, "mean_pool": args.mean_pool,
         "cl": args.cl, "clip": args.clip, "description": args.desc
         },
        {"hparam/MAE": np.mean(mae), "hparam/MAPE": np.mean(mape), "hparam/RMSE": np.mean(rmse),
         "hparam/loss": losses_mean / len(test_loader), "hparam/train_reg_loss": mtrain_loss if not args.inference else 0,}
    )
    writer.close()
    print("Overall results, MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}".format(np.mean(mae), np.mean(mape), np.mean(rmse)))
    print("Experiments finished...Checkout the results at:", log_dir)
    if args.optuna:
        serializable_dict = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
        trial.set_user_attr(key="model", value=serializable_dict)
        trial.set_user_attr(key="log_dir", value=log_dir)
    return mvalid_rmse if not args.inference else np.mean(rmse)


def save_model_callback(study, trial):
    if study.best_trial == trial:
        joblib.dump(study, f'./runs/study_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl')
        torch.save(trial.user_attrs['model'], f'{trial.user_attrs["log_dir"]}/best_model.pth')

from optuna.trial import TrialState
class ResumeFailedTrialsCallback:
    def __init__(self, max_retry=3):
        self.max_retry = max_retry
        self.retry_count = {}

    def __call__(self, study, trial):
        try:
            if trial.state == TrialState.FAIL:
                trial_id = trial.number
                self.retry_count.setdefault(trial_id, 0)
                if self.retry_count[trial_id] < self.max_retry:
                    self.retry_count[trial_id] += 1
                    print(f"Retrying failed trial {trial_id} (Attempt {self.retry_count[trial_id]})")
                    new_trial = study.ask(fixed_params=trial.params)
                    print(f"Resuming trial {trial_id} from the study {study._study_id} as new trial {new_trial.number}")
                    study.tell(new_trial, state=TrialState.RUNNING)
        except Exception as _:
            pass


if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    N_TRIALS = 500
    N_JOBS = 2 #TODO:  OPT
    if args.optuna:
        sampler = optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True)
        # old_study = optuna.load_study(study_name="no-name-91a30809-34bd-450c-9376-2ddd3bbf4ad0", storage="sqlite:///db_2024-09-11_03-44-13.sqlite3")
        new_study = optuna.create_study(
            direction="minimize",
            storage=f'sqlite:///db_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.sqlite3', 
            sampler=sampler, 
            # study_name="no-name-0c0687ce-f536-4e02-959d-a370a3b5305b", 
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
            # load_if_exists=True
        )
        # for trial in old_study.trials:
        #     if trial.state == optuna.trial.TrialState.COMPLETE:
        #         new_study.add_trial(trial)
        # resume_callback = ResumeFailedTrialsCallback(max_retry=1)
        new_study.optimize(main_pems04, n_jobs=N_JOBS, n_trials=N_TRIALS, show_progress_bar=True, callbacks=[save_model_callback], catch=(Exception,))
        print('Number of finished trials:', len(new_study.trials))
        print('Best trial:')
        trial = new_study.best_trial
        print('  Value: ', trial.value)
        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
    else:
        from dataclasses import dataclass
        from uuid import uuid4


        @dataclass
        class Trial:
            number: int | str
            params: dict


        trial = Trial(str(uuid4()), {})
        runners = {
            "datasets/PEMS04": main_pems04,
            args.data: main_pems04,
            # "datasets/METRLA": main,
        }
        runners[args.data](trial)
