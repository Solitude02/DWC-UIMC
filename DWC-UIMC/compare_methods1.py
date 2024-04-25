import numpy as np
from util import mv_dataset, read_mymat, build_ad_dataset, process_data, get_validation_set, \
    mv_tabular_collate
from scipy.spatial.distance import cdist
from icecream import ic
ic.disable()


def get_samples(x, y, sn, train_index, test_index, n_sample, k, if_mean=False, reg_param=1e-6):
    view_num = len(x)
    data_num = x[0].shape[0]

    print("使用均值填充缺失视图")
    # 计算每个视图可用样本的均值
    for v in range(view_num):
        # 筛选可用样本
        x_v_mean = np.mean(x[v][sn[:,v] == 1], axis=0)
        # 所有缺失样本用均值填充
        x[v][sn[:,v] == 0] = x_v_mean
    
    # 将样本重复n_sample次
    x_train = [np.repeat(x[_][train_index], n_sample, axis=0) for _ in range(view_num)]
    y_train = np.repeat(y[train_index], n_sample, axis=0)
    Sn_train = np.repeat(sn[train_index], n_sample, axis=0)
    x_test = [x[_][test_index] for _ in range(view_num)]
    y_test = y[test_index]

    return x_train, y_train, x_test, y_test, Sn_train
    
if __name__ == '__main__':
    dataset_name = 'handwritten0.mat'
    view_num = 6
    # dataset_name = 'BRCA.mat'
    # view_num = 3
    missing_rate = 0.5
    X, Y, Sn = read_mymat('./data/', dataset_name, ['X', 'Y'], missing_rate)
    partition = build_ad_dataset(Y, p=0.8, seed=999)

    X = process_data(X, view_num)
    print(partition['train'].shape, partition['test'].shape)
    X_train, Y_train, X_test, Y_test, Sn_train=get_samples(X, Y, Sn, partition['train'], partition['test'], 5, 10)
    print(X_train[0].shape, Y_train.shape, X_test[0].shape, Y_test.shape, Sn_train.shape)