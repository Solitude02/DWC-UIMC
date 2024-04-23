import numpy as np
import scipy.io as scio
import scipy.sparse as scsp
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from numpy.random import randint
import numpy.ma as npm
import pandas as pd
from get_sn import get_sn, load_Sn

import copy
from sklearn.preprocessing import StandardScaler


transform = transforms.Compose([
    transforms.ToTensor(),
])


def read_mymat(path, name, sp, missrate, sparse=False):
    # 读取数据集
    '''
    :param path: 数据集路径
    :param name: 数据集名称
    :param sp: 从数据集中读取哪些字段，如['X', 'Y']
    :param missrate: 缺失率
    :param sparse: 一个布尔参数，用于指示是否将数据集中的数据转换为稀疏矩阵
    :return: X, Y, Sn
    '''
    # 路径
    mat_file = path + name
    f = scio.loadmat(mat_file)


    if 'Y' in sp:
        # 读取gt字段作为Y
        if (name == 'handwritten0.mat') or (name == 'BRCA.mat') or (name == 'ROSMAP.mat'):
            Y = (f['gt']).astype(np.int32)
        else:
            Y = (f['gt']-1).astype(np.int32)
    else:
        Y = None


    if 'X' in sp:
        # 读取X字段，并将其重塑为一维数组
        Xa = f['X']
        # 将Xa重塑为一个一维数组，其长度等于Xa的第二个维度的大小。
        Xa = Xa.reshape(Xa.shape[1], )
        # 此时对于handwritten0.mat数据集，Xa的形状为(6,)，代表6个视图（特征集）
        X = []
        if sparse:
            for x in Xa:
                # 将Xa中的每个元素转换为稀疏矩阵
                # scsp.csc_matrix(x)将x转换为稀疏矩阵的一种形式，称为压缩稀疏列（CSC）格式
                # 例如，考虑以下稀疏矩阵：
                # 0 0 1
                # 2 0 0
                # 0 3 0
                # 在CSC格式中，它将被表示为：
                # values: [2, 3, 1]
                # row_indices: [1, 2, 0]
                # col_ptrs: [0, 1, 2, 3]，其中3代表values中的元素数
                # .toarray()方法将稀疏矩阵转换为一个密集的numpy数组
                X.append(scsp.csc_matrix(x).toarray().astype(np.float64))
        else:
            for x in Xa:
                # 将Xa中的每个元素转换为float64类型的矩阵
                X.append(x.astype(np.float64))
    else:
        X = None
    n_sample = len(X[0][0]) #样本数，对于handwritten0.mat数据集，n_sample=2000
    n_view = len(X) # 视图数=特征集数
    # Sn = get_sn(n_view, n_sample, missrate).astype(np.float32) # 生成缺失索引矩阵

    # 改成从文件中读取缺失索引矩阵
    # 读取对应缺失率的缺失索引矩阵
    if name == 'handwritten0.mat':
        sn_name = "sh_" + str(missrate)[2:]
    elif name == 'BRCA.mat':
        sn_name = "BRCA_sh_" + str(missrate)[2:]
    
    # 读取缺失索引矩阵，缺失率为0时，Sn为全1矩阵
    if missrate == 0:
        Sn = np.ones((n_sample, n_view))
    else:
        Sn = load_Sn(sn_name).astype(np.float32)

    for i in range(n_view):
        X[i] = X[i].T
    return X, Y, Sn
    '''
    X: 数据集的特征集，形状为(视图数, 样本数, 特征数)
    Y: 数据集的标签，形状为(样本数,)
    Sn: 数据集的缺失索引矩阵，形状为(样本数, 视图数)
    '''



def build_ad_dataset(Y, p, seed=999):
    '''
    将原始的多类别多视图数据集转换为异常检测数据集（用构建异常检测数据集的方法划分训练集和测试集）
    :param seed: 随机数种子
    :param p: 训练集的正常样本比例proportion of normal samples for training
    :param neg_class_id: 该类用作负类（离群值）the class used as negative class (outliers)
    :param Y: 原始类索引the original class indexes
    :return:
    '''
    np.random.seed(seed=seed)
    Y = np.squeeze(Y) # 压缩Y，去掉维度为1的轴,确保维度正确
    Y_idx = np.array([x for x in range(len(Y))]) # Y的索引
    num_normal_train = np.int_(np.ceil(len(Y_idx) * p)) # 训练集中正常样本的数量
    # 随机选择训练集中正常样本的索引（用于训练集）
    train_idx_idx = np.random.choice(len(Y_idx), num_normal_train, replace=False) # 在Y_idx中随机选择num_normal_train个索引
    train_idx = Y_idx[train_idx_idx] # 训练集中正常样本的索引
    test_idx = np.array(list(set(Y_idx.tolist()) - set(train_idx.tolist()))) # 测试集中的索引
    partition = {'train': train_idx, 'test': test_idx} # 划分数据集
    return partition



def get_validation_set_Sn(X_train, Sn_train, Y_train, val_p=0.1):
    tot_num = X_train[0].shape[0]
    perm = np.random.permutation(tot_num)
    val_idx = perm[:np.int_(np.ceil(tot_num * val_p))]
    train_idx = perm[np.int_(np.ceil(tot_num * val_p)):]
    val_set = [X_train[i][val_idx] for i in range(len(X_train))]
    val_train_set = [X_train[i][train_idx] for i in range(len(X_train))]
    val_train_set_Y = Y_train[train_idx]
    val_val_set_Y = Y_train[val_idx]
    val_train_set_Sn = Sn_train[train_idx]
    val_val_set_Sn = Sn_train[val_idx]
    return val_train_set, val_set, val_train_set_Sn, val_val_set_Sn, val_train_set_Y, val_val_set_Y


def get_validation_set(Y, p, seed=999):
    np.random.seed(seed=seed)
    Y = np.squeeze(Y)
    Y_idx = np.array([x for x in range(len(Y))])
    num_normal_train = np.int_(np.ceil(len(Y_idx) * p))
    train_idx_idx = np.random.choice(len(Y_idx), num_normal_train, replace=False)
    train_idx = Y_idx[train_idx_idx]
    test_idx = np.array(list(set(Y_idx.tolist()) - set(train_idx.tolist())))
    partition = {'train': train_idx, 'val': test_idx}
    return partition



def process_data(X, n_view):
    # 数据预处理
    eps = 1e-10
    if (n_view == 1): 
        # 如果视图数为1,则将X转换为一个numpy数组
        # StandardScaler()用于标准化数据
        X = StandardScaler().fit_transform(X)
    else:
        # 如果视图数大于1,则将X转换为一个列表
        X = [StandardScaler().fit_transform(X[i]) for i in range(n_view)]
    return X


class partial_mv_dataset(Dataset):
    # 用于处理部分多视图数据集
    def __init__(self, data, Sn, Y):
        '''
        :param data: Input data is a list of numpy arrays
        '''
        self.data = data
        self.Y = Y
        self.Sn = Sn

    def __getitem__(self, item):
        datum = [self.data[view][item][np.newaxis, :] for view in range(len(self.data))]
        Y = self.Y[item]
        Sn = self.Sn[item].reshape(1, len(self.Sn[item]))
        return [torch.from_numpy(datum[view]) for view in range(len(self.data))], torch.from_numpy(Sn), torch.from_numpy(Y)

    def __len__(self):
        return self.data[0].shape[0]


class mv_dataset(Dataset):
    # 用于处理完整多视图数据集
    def __init__(self, data, Y):
        '''
        :param data: Input data is a list of numpy arrays
        '''
        self.data = data
        self.Y = Y

    def __getitem__(self, item):
        datum = [self.data[view][item][np.newaxis, :] for view in range(len(self.data))]
        Y = self.Y[item]
        return [torch.from_numpy(datum[view]) for view in range(len(self.data))], torch.from_numpy(Y)

    def __len__(self):
        return self.data[0].shape[0]

def partial_mv_tabular_collate(batch):
    # 用于处理部分多视图数据集

    new_batch = [[] for _ in range(len(batch[0][0]))]
    new_label = []
    new_Sn = []
    for y in range(len(batch)):
        cur_data = batch[y][0]
        Sn_data = batch[y][1]
        label_data = batch[y][2]
        for x in range(len(batch[0][0])):
            new_batch[x].append(cur_data[x])
        new_Sn.append(Sn_data)
        new_label.append(label_data)
    return [torch.cat(new_batch[i], dim=0) for i in range(len(batch[0][0]))], torch.cat(new_Sn, dim=0), torch.cat(new_label, dim=0)

def mv_tabular_collate(batch):
    new_batch = [[] for _ in range(len(batch[0][0]))]
    new_label = []
    for y in range(len(batch)):
        cur_data = batch[y][0]
        label_data = batch[y][1]
        for x in range(len(batch[0][0])):
            new_batch[x].append(cur_data[x])
        new_label.append(label_data)
    return [torch.cat(new_batch[i], dim=0) for i in range(len(batch[0][0]))],  torch.cat(new_label, dim=0)

def tensor_intersection(x, y):
    return torch.tensor(list(set(x.tolist()).intersection(set(y.tolist()))))



