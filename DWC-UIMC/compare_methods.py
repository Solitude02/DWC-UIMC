import numpy as np
from util import mv_dataset, read_mymat, build_ad_dataset, process_data, get_validation_set, \
    mv_tabular_collate
from scipy.spatial.distance import cdist

# 定义一个正则化参数，用于多元高斯分布的协方差矩阵的正则化，以防止过拟合。
reg_param  = 1e-3


def get_samples(x, y, sn, train_index, test_index, n_sample, k, if_mean=False, reg_param=1e-6):
    view_num = len(x)
    data_num = x[0].shape[0]

    # 计算每个视图中所有样本之间的欧氏距离
    print("计算距离")
    dist_all_set = [cdist(x[i], x[i], 'euclidean') for i in range(view_num)]

    # 构造一个 view_num*view_num 的列表，每个元素是一个数组，表示第i和j视图中都存在的所有样本的索引
    dismiss_view_index = [[np.array([]) for __ in range(view_num)] for _ in range(view_num)]
    # 计算每个视图中都存在的样本索引
    for i in range(view_num - 1):
        for j in range(i + 1, view_num):
            # i和j视图中都存在的样本索引，即完整样本
            sn_temp = sn[:, [i, j]]
            sn_temp_sum = np.sum(sn_temp, axis=1)
            sn_temp_sum[test_index] = 0
            dismiss_view_index[i][j] = dismiss_view_index[j][i] = np.where(sn_temp_sum == 2)[0]

    print("使用距离分配权重填充训练集中的缺失视图")
    sn_train = sn[train_index]
    x_train = [x[v][train_index] for v in range(view_num)]
    y_train = y[train_index]

    # step3: 筛选完整样本
    x_train_dissmiss_index = np.where(np.sum(sn_train, axis=1) == view_num)[0]
    x_complete = [x_train[_][x_train_dissmiss_index] for _ in range(view_num)]
    y_complete = y_train[x_train_dissmiss_index]
    sn_complete = sn_train[x_train_dissmiss_index]

    # step4: 填充不完整视图
    x_train_miss_index = np.where(np.sum(sn_train, axis=1) < view_num)[0] # 找到训练集中缺失的视图
    x_incomplete = [np.repeat(x_train[_][x_train_miss_index], n_sample, axis=0) for _ in range(view_num)]
    y_incomplete = np.repeat(y_train[x_train_miss_index], n_sample, axis=0)
    sn_incomplete = np.repeat(sn_train[x_train_miss_index], n_sample, axis=0)
    index = 0
    for i in x_train_miss_index: # 遍历缺失视图的索引
        y_i = y_train[i][0] # 获取标签
        miss_view_index = np.nonzero(sn_train[i] == 0)[0] # 获取缺失视图的索引
        # np.nonzero(sn_train[i] == 0)返回sn_train[i]中为0的索引
        # nonzero的作用是返回数组a中非零元素的索引值数组
        for v in miss_view_index:
            # 计算样本与其他样本的距离
            dist_v = dist_all_set[v][i]
            # 根据距离计算权重
            weights = np.exp(-dist_v / np.max(dist_v)) # 距离越近，权重越大
            # 对权重进行归一化
            weights = weights / np.sum(weights)
            # 随机采样
            chosen_indices = np.random.choice(data_num, n_sample, p=weights)
            x_incomplete[v][index * n_sample:(index + 1) * n_sample] = x[v][chosen_indices]
        index += 1

    x_train = [np.concatenate((x_complete[_], x_incomplete[_]), axis=0) for _ in range(view_num)]
    x_train = process_data(x_train, view_num)
    y_train = np.concatenate((y_complete, y_incomplete), axis=0)
    Sn_train = np.concatenate((sn_complete, sn_incomplete), axis=0)

    print("基于距离填充测试集中的缺失视图")
    # 填充测试集时，只考虑训练集中存在的视图
    sn_test = sn[test_index]
    x_test_dissmiss_index = np.where(np.sum(sn_test, axis=1) == view_num)[0]
    if if_mean:
        # 使用多个采样点的均值填充缺失视图
        x_test = [x[_][test_index][x_test_dissmiss_index] for _ in range(view_num)]
        y_test = y[test_index][x_test_dissmiss_index]
    else:
        # 使用所有采样点填充缺失视图
        x_test = [np.repeat(x[_][test_index][x_test_dissmiss_index], n_sample, axis=0) for _ in range(view_num)]
        y_test = np.repeat(y[test_index][x_test_dissmiss_index], n_sample, axis=0)

    for i in test_index.flat: # 遍历测试集索引
        if if_mean:
            # view_num * (1, dim,)
            x_i = [np.expand_dims(x[_][i], axis=0) for _ in range(view_num)]
            # (1, 1)
            y_i = np.expand_dims(y[i], axis=0)
        else:
            # view_num * (n_sample, dim,)
            x_i = [np.repeat(np.expand_dims(x[_][i], axis=0), n_sample, axis=0) for _ in range(view_num)]
            # (n_sample, 1,)
            y_i = np.repeat(np.expand_dims(y[i], axis=0), n_sample, axis=0)

        sn_temp = sn[i] # sn[i]：(view_num,)，记录第i个样本的视图数量
        x_miss_view_index = np.nonzero(sn_temp == 0)[0] # 缺失视图索引，形状为(缺失视图数,)
        x_dismiss_view_index = np.nonzero(sn_temp)[0] # 存在视图索引，形状为(存在视图数,)

        # 填充不完整样本
        if x_miss_view_index.shape[0] != 0: # 如果存在缺失视图（缺失视图索引的行数不为0）
            for j in x_miss_view_index.flat: # 遍历拥有缺失视图的样本
                # 获取x在第j个视图上的邻居索引
                neighbors_index_temp = np.array([], dtype=np.int_)
                for jj in x_dismiss_view_index.flat: #遍历拥有两个以上视图的样本，只要同时拥有j和另外一个视图即可
                    dismiss_view_index_temp = dismiss_view_index[j][jj]  # 第j和jj个视图都存在的样本索引
                    dist_temp = np.full(data_num, np.inf) # 初始化距离
                    dist_temp[dismiss_view_index_temp] = dist_all_set[jj][i, dismiss_view_index_temp] # 计算距离
                    dist_temp[i] = np.inf  # 将自己与自己的距离设置为无穷大，排除自己作为邻居
                    nearest_index_temp = np.argpartition(dist_temp, k)[:k] # 获取最近的k个邻居
                    # np.argpartition(a, k)会返回一个新的数组，其中前 k 个元素是 dist_temp 中最小的 k 个元素的索引
                    neighbors_index_temp = np.unique(
                        np.concatenate((neighbors_index_temp, nearest_index_temp), ))  # 连接所有存在的视图

                # 从近邻集中基于距离随机采样填充缺失视图
                for v in x_miss_view_index.flat:
                    rng = np.random.default_rng() # 随机数生成器
                    x_i[v] = x[v][neighbors_index_temp]
                    y_i = y[neighbors_index_temp]
                    # 计算概率
                    dist_temp_nonzero = dist_temp[neighbors_index_temp].copy()
                    dist_temp_nonzero[dist_temp_nonzero == 0] = 1e-10  # 将0值替换为一个非常小的值
                    probabilities = 1/dist_temp_nonzero
                    # 归一化概率
                    probabilities /= probabilities.sum()
                    selected_indices = rng.choice(x_i[v].shape[0], size=n_sample, replace=True, p=probabilities)
                    x_i[v] = x_i[v][selected_indices] # 随机选取n_sample个样本

                # # 从近邻集中随机采样填充缺失视图
                # for _ in range(n_sample):
                #     sampled_index = np.random.choice(neighbors_index_temp) # 从邻居索引中随机选择一个样本
                #     x_i[j][_] = x[j][sampled_index] # 使用选中的样本的视图数据填充缺失视图

            x_test = [np.concatenate((x_test[_], x_i[_]), axis=0) for _ in range(view_num)] # 连接所有视图
            y_test = np.concatenate((y_test, y_i), axis=0) # 连接所有标签
    x_test = process_data(x_test, view_num)

    return x_train, y_train, x_test, y_test, Sn_train


if __name__ == '__main__':
    dataset_name = 'handwritten0.mat'
    view_num = 6
    missing_rate = 0.5
    X, Y, Sn = read_mymat('./data/', dataset_name, ['X', 'Y'], missing_rate)
    partition = build_ad_dataset(Y, p=0.8, seed=999)

    X = process_data(X, view_num)
    X_train, Y_train, X_test, Y_test, Sn_train=get_samples(X, Y, Sn, partition['train'], partition['test'], 5, 10)

