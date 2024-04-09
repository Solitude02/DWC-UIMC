import numpy as np
from scipy.spatial.distance import cdist
from util import read_mymat, process_data, build_ad_dataset
import matplotlib.pyplot as plt

# 导入原来的方法和新的方法
from select_k_neighbors import get_samples as get_samples_gaussian
from compare_methods import get_samples as get_samples_distance

def get_samples(x, y, sn, train_index, test_index, n_sample, k, if_mean=False, reg_param=1e-6):
    # 如果不需要进行处理，直接返回原始数据
    x_train = [x[v][train_index] for v in range(len(x))]
    y_train = y[train_index]
    x_test = [x[v][test_index] for v in range(len(x))]
    y_test = y[test_index]

    x_train = process_data(x_train, len(x))
    x_test = process_data(x_test, len(x))

    return x_train, y_train, x_test, y_test, sn[train_index]

import os

def compare_fill_methods(dataset_name, view_num, missing_rate):
    data_dir = "D:/毕设/DWC-UIMC/DWC-UIMC/data/"
    X, Y, Sn = read_mymat(data_dir, dataset_name, ['X', 'Y'], missing_rate)
    partition = build_ad_dataset(Y, p=0.8, seed=999)

    X = process_data(X, view_num)
    X_train, Y_train, X_test, Y_test, Sn_train = get_samples(X, Y, Sn, partition['train'], partition['test'], 5, 10)

    # 使用多次从多元高斯分布采样的方法填充不完整视图
    X_train_gaussian, _, X_test_gaussian, _, _ = get_samples_gaussian(X, Y, Sn, partition['train'], partition['test'], 5, 10, if_mean=False)

    # 使用基于欧氏距离加权平均的方法填充不完整视图
    X_train_euclidean, _, X_test_euclidean, _, _ = get_samples_distance(X, Y, Sn, partition['train'], partition['test'], 5, 10, if_mean=True)

    # 可视化比较填充效果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(view_num):
        axes[0, i].imshow(X_train[i][0].reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f'View {i+1} Original')
        axes[0, i].axis('off')

        axes[1, i].imshow(X_train_gaussian[i][0].reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f'View {i+1} Gaussian Sampling')
        axes[1, i].axis('off')

        axes[2, i].imshow(X_train_euclidean[i][0].reshape(28, 28), cmap='gray')
        axes[2, i].set_title(f'View {i+1} Euclidean Weighted Average')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dataset_name = 'handwritten0.mat'
    view_num = 6
    missing_rate = 0.5
    compare_fill_methods(dataset_name, view_num, missing_rate)
