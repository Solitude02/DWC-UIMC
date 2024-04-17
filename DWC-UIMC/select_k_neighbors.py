import numpy as np
from util import mv_dataset, read_mymat, build_ad_dataset, process_data, get_validation_set, \
    mv_tabular_collate
from scipy.spatial.distance import cdist

# 定义一个正则化参数，用于多元高斯分布的协方差矩阵的正则化，以防止过拟合。
reg_param  = 1e-3


def get_samples(x, y, sn, train_index, test_index, n_sample, k, if_mean=False, reg_param=1e-6):
    # 用于获取处理过的样本数据。
    # x: 数据集: view_num * (dataset_num, dim,)
    # y: 标签: (dataset_num,)
    # sn: 缺失索引矩阵: (dataset_num, view_num,)
    # 如果 sn 矩阵中的某个元素为1，那么表示对应的视图是完整的；如果为0，那么表示对应的视图是不完整的。
    # train_index: 训练集索引: (train_num,)
    # test_index: 测试集索引: (test_num,)
    # n_sample: 采样频率，用于实现“多次采样填充缺失视图”
    # k: 邻居数量
    # if_mean: 是否使用均值填充缺失视图
    # reg_param: 正则化参数

    view_num = len(x) # 视图数量
    data_num = x[0].shape[0]  # 第一个视图的样本数，即样本数

    # 计算每个视图中所有样本之间的欧氏距离
    print("计算距离")
    # cdist这个函数会忽略任何缺失的数据，并且将它们视为零。这可能会导致距离计算的结果不准确。
    dist_all_set = [cdist(x[i], x[i], 'euclidean') for i in range(view_num)]
    # 构造一个 view_num*view_num 的列表，每个元素是一个数组，表示第i和j视图中都存在的所有样本的索引
    dismiss_view_index = [[np.array([]) for __ in range(view_num)] for _ in range(view_num)]
    
    # 计算每个视图中都存在的样本索引
    for i in range(view_num - 1):
        for j in range(i + 1, view_num):
            # i和j视图中都存在的样本索引，即完整样本
            sn_temp = sn[:, [i, j]]  # sn_temp: (dataset_num, 2)选取sn的第i和j列，即第i和j视图的缺失数据索引
            sn_temp_sum = np.sum(sn_temp, axis=1) # 求和，得到每个样本的缺失视图数量
            sn_temp_sum[test_index] = 0  # 掩盖测试集样本，即”测试集样本不参与计算“
            dismiss_view_index[i][j] = dismiss_view_index[j][i] = np.where(sn_temp_sum == 2)[0]
            # sn_temp_sum == 2 表示两个视图都存在
            '''
            sn = np.array([
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1]
            ])  # 形状为 (4, 3)
            test_index = np.array([2])  #2是测试集索引
            对于视图0和视图1，都存在的样本索引为[0, 2]
            sn_temp = np.array([
                [1, 1],
                [1, 0],
                [1, 1],
                [0, 1]
            ])  # 形状为 (4, 2)
            sn_temp_sum = np.array([2, 1, 0, 1])  # 形状为 (4,)
            dismiss_view_index[0][1] = dismiss_view_index[1][0] = np.array([0])  # 形状为 (1,)
            '''

    print("使用多元高斯分布填充训练集中的缺失视图")
    # step1: 获取训练集
    sn_train = sn[train_index] # sn_train：(train_num, view_num)，记录第train_num个样本的缺失视图
    x_train = [x[v][train_index] for v in range(view_num)] # x_train: 
    y_train = y[train_index] # y_train:
    # step2: 在每个视图上获取每个类的样本点
    class_num = np.max(y) + 1
    means, covs, num = dict(), dict(), dict()
    for v in range(view_num):
        present_index = np.where(sn_train[:, v] == 1)[0]
        means_v, covs_v, num_v = [], [], []
        for c in range(class_num):
            present_index_class = np.where(y_train[present_index] == c)[0]
            means_v.append(np.mean(x_train[v][present_index_class], axis=0))
            covs_v.append(np.cov(x_train[v][present_index_class], rowvar=0))
            num_v.append(present_index_class.shape[0])
        means[v], covs[v], num[v] = means_v, covs_v, num_v
        # means形式为：{第v个视图：[第c个类的均值，第c+1个类的均值，...], 第v+1个视图：[第c个类的均值，第c+1个类的均值，...], ...}

    # step3: 筛选完整样本
    x_train_dissmiss_index = np.where(np.sum(sn_train, axis=1) == view_num)[0]
    x_complete = [x_train[_][x_train_dissmiss_index] for _ in range(view_num)]
    y_complete = y_train[x_train_dissmiss_index]
    sn_complete = sn_train[x_train_dissmiss_index]
    # print(f"完整样本数量：{len(y_complete)}")


    # step4: 填充不完整视图
    x_train_miss_index = np.where(np.sum(sn_train, axis=1) < view_num)[0] # 找到训练集中缺失的视图
    x_incomplete = [np.repeat(x_train[_][x_train_miss_index], n_sample, axis=0) for _ in range(view_num)]
    y_incomplete = np.repeat(y_train[x_train_miss_index], n_sample, axis=0)
    sn_incomplete = np.repeat(sn_train[x_train_miss_index], n_sample, axis=0)
    # print(f"不完整样本数量（填充前）：{len(y_train[x_train_miss_index])}")
    # print(f"不完整样本数量（填充后）：{len(y_incomplete)}")

    index = 0
    for i in x_train_miss_index: # 遍历拥有缺失视图的样本
        y_i = y_train[i][0] # 获取标签
        miss_view_index = np.nonzero(sn_train[i] == 0)[0] # 获取缺失视图的索引
        # np.nonzero(sn_train[i] == 0)返回sn_train[i]中为0的索引
        # nonzero的作用是返回数组a中非零元素的索引值数组
        for v in miss_view_index:
            rng = np.random.default_rng()
            cov = covs[v][y_i] + np.eye(len(covs[v][y_i])) * reg_param  # 添加正则化参数以确保非单位性
            L = np.linalg.cholesky(cov)  # Cholesky分解，得到下三角矩阵
            samples_v = rng.normal(size=(n_sample, len(cov))) @ L.T + means[v][y_i]  # 生成服从多元正态分布的样本

            x_incomplete[v][index * n_sample:(index + 1) * n_sample] = samples_v  # 填充缺失视图
        index += 1

    x_train = [np.concatenate((x_complete[_], x_incomplete[_]), axis=0) for _ in range(view_num)]
    x_train = process_data(x_train, view_num)
    y_train = np.concatenate((y_complete, y_incomplete), axis=0)
    Sn_train = np.concatenate((sn_complete, sn_incomplete), axis=0)
    # print(f"最终训练集样本数量：{len(y_train)}")

    print("基于多元高斯分布填充测试集中的缺失视图")
    sn_test = sn[test_index]
    x_test_dissmiss_index = np.where(np.sum(sn_test, axis=1) == view_num)[0]
    # x_test_incomplete_index = np.where(np.sum(sn_test, axis=1) < view_num)[0]
    # print(f"完整样本数量：{len(x_test_dissmiss_index)}")
    # print(f"不完整样本数量（填充前）：{len(x_test_incomplete_index)}")

    '''在训练集中，我们通常不对完整的样本进行重复，因为这可能会导致过拟合。
    过拟合是指模型在训练数据上表现得过于优秀，以至于无法很好地泛化到新的、未见过的数据上。
    如果我们对训练集中的样本进行重复，那么模型可能会过度学习这些重复的样本，从而无法很好地处理新的数据。

    然而，在测试集中，我们可能需要对完整的样本进行重复，以便更好地评估模型的性能。
    这是因为，测试集的目的是模拟真实世界的数据分布，而在真实世界中，某些样本可能会出现多次。
    通过在测试集中重复这些样本，我们可以更准确地评估模型在处理这些常见样本时的性能。
    '''
    if if_mean:
        # 只包含完整视图的样本
        x_test = [x[_][test_index][x_test_dissmiss_index] for _ in range(view_num)]
        y_test = y[test_index][x_test_dissmiss_index]
    else:
        # 重复n_sample次包含完整视图的样本
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

        sn_temp = sn[i] # sn[i]：(view_num,)，记录第i个样本的视图数量，sn_temp == 0则该视图缺失
        x_miss_view_index = np.nonzero(sn_temp == 0)[0] # 缺失视图索引，[缺失视图1，缺失视图2，...]
        x_dismiss_view_index = np.nonzero(sn_temp)[0] # 完整视图索引，[完整视图1，完整视图2，...]

        # 填充不完整样本
        if x_miss_view_index.shape[0] != 0: # 如果存在缺失视图（缺失视图索引的行数不为0）
            for j in x_miss_view_index.flat: # 遍历缺失视图
                # 获取x在第j个视图上的邻居索引
                neighbors_index_temp = np.array([], dtype=np.int_)
                for jj in x_dismiss_view_index.flat: 
                    #遍历拥有两个以上视图的样本，只要同时拥有j和另外一个视图即可
                    dismiss_view_index_temp = dismiss_view_index[j][jj]  # 第j和jj个视图都存在的样本索引
                    dist_temp = np.full(data_num, np.inf) # 初始化距离
                    dist_temp[dismiss_view_index_temp] = dist_all_set[jj][i, dismiss_view_index_temp] # 计算距离
                    nearest_index_temp = np.argpartition(dist_temp, k)[:k] # 获取最近的k个邻居
                    # np.argpartition(a, k)会返回一个新的数组，其中前 k 个元素是 dist_temp 中最小的 k 个元素的索引
                    neighbors_index_temp = np.unique(
                        np.concatenate((neighbors_index_temp, nearest_index_temp), ))  # 连接所有存在的视图

                x_neighbors_temp = x[j][neighbors_index_temp] # 补全第j个视图的邻居集
                mean = np.mean(x_neighbors_temp, axis=0) # 计算均值
                cov = np.cov(x_neighbors_temp, rowvar=0) # 计算协方差
                rng = np.random.default_rng() # 随机数生成器
                cov = cov + np.eye(len(cov)) * reg_param # 添加正则化参数
                L = np.linalg.cholesky(cov) # Cholesky分解
                x_samples_temp = rng.normal(size=(n_sample, len(cov))) @ L.T + mean # 生成服从多元正态分布的样本
                x_i[j] = x_samples_temp # 填充缺失视图

            x_test = [np.concatenate((x_test[_], x_i[_]), axis=0) for _ in range(view_num)] # 连接所有视图
            y_test = np.concatenate((y_test, y_i), axis=0) # 连接所有标签
    # print(f"不完整样本数量（填充后）：{len(y_test) - len(x_test_dissmiss_index)}")
    # print(f"最终测试集样本数量：{len(y_test)}")

    x_test = process_data(x_test, view_num)

    return x_train, y_train, x_test, y_test, Sn_train
    '''
    x_train: 训练集样本数据，形状为 view_num * (train_num, dim)
    y_train: 训练集标签，形状为 (train_num,)
    x_test: 测试集样本数据，形状为 view_num * (test_num, dim)
    y_test: 测试集标签，形状为 (test_num,)
    Sn_train: 训练集中的缺失视图索引，形状为 (train_num, view_num)
    '''


if __name__ == '__main__':
    # dataset_name = 'handwritten0.mat'
    # view_num = 6
    dataset_name = 'BRAC.mat'
    view_num = 3
    missing_rate = 0.3
    X, Y, Sn = read_mymat('./data/', dataset_name, ['X', 'Y'], missing_rate)
    partition = build_ad_dataset(Y, p=0.8, seed=999)

    X = process_data(X, view_num)
    print(partition['train'].shape, partition['test'].shape)
    X_train, Y_train, X_test, Y_test, Sn_train=get_samples(X, Y, Sn, partition['train'], partition['test'], 5, 10)
    print(X_train[0].shape, Y_train.shape, X_test[0].shape, Y_test.shape, Sn_train.shape)

