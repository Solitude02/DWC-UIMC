import numpy as np
from scipy.spatial.distance import cdist
from select_k_neighbors import get_samples as get_samples_gaussian  # 假设这里是您的 get_samples 方法的导入
from compare_methods import get_samples as get_samples_distance  # 假设这里是您的 get_samples 方法的导入
from util import read_mymat, build_ad_dataset, process_data, build_ad_dataset

# 假设完整的数据集为 complete_data，形状为 (view_num, dataset_num, dim)
# 假设使用 get_sn 方法生成缺失索引矩阵 sn，形状为 (dataset_num, view_num)
# 假设 train_index 和 test_index 是训练集和测试集的索引
def compare_methods(X, Y, Sn, train_index, test_index,method):
    # # 查看索引类型与形状
    # print('train_index:',type(train_index),train_index.shape)
    # print('test_index:',type(test_index),test_index.shape)
    # print('train_index dtype:', train_index.dtype)
    # print('test_index dtype:', test_index.dtype)

    if method == 'gaussian':
        print("基于高斯分布的填充方法：")
        get_samples = get_samples_gaussian
    else:
        print("基于距离的填充方法：")
        get_samples = get_samples_distance
    # 计算填充效果
    X_train, Y_train, X_test, Y_test, Sn_train=get_samples(X, Y, Sn, partition['train'], partition['test'], 1, 10)

    # 计算与原始数据集的差距
    # 获取训练集的数据
    X_train_origin = [x[train_index, :] for x in X]
    # 获取测试集的数据
    X_test_origin = [x[test_index, :] for x in X]
    # # 查看训练集的数据形状
    # print('X_train_origin:',[x.shape for x in X_train_origin])
    # # 查看填充得到的训练集的数据形状
    # print('X_train:',[x.shape for x in X_train])

    # 训练集：
    # 计算每个视图的填充效果,通过均方误差表示
    filling_diff_per_view = [np.mean(np.square(x_origin - x_train)) for x_origin, x_train in zip(X_train_origin, X_train)]
    # 打印每个视图的填充效果
    for i, diff in enumerate(filling_diff_per_view):
        print(f'Training Filling difference for view {i}: {diff}')
    
    # 测试集：
    # 计算每个视图的填充效果,通过均方误差表示
    filling_diff_per_view = [np.mean(np.square(x_origin - x_test)) for x_origin, x_test in zip(X_test_origin, X_test)]
    # 打印每个视图的填充效果
    for i, diff in enumerate(filling_diff_per_view):
        print(f'Test Filling difference for view {i}: {diff}')


if __name__ == '__main__':
    dataset_name = 'handwritten0.mat'
    view_num = 6
    missing_rate = 0.3
    X, Y, Sn = read_mymat('./data/', dataset_name, ['X', 'Y'], missing_rate)
    partition = build_ad_dataset(Y, p=0.8, seed=999)

    X = process_data(X, view_num)
    # 测试 compare_methods 方法
    compare_methods(X, Y, Sn, partition['train'], partition['test'],'gaussian')
    compare_methods(X, Y, Sn, partition['train'], partition['test'],'distance')
