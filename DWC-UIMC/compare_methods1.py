# 基于距离形成近邻集加权平均，只进行一次填充
import numpy as np
from util import mv_dataset, read_mymat, build_ad_dataset, process_data, get_validation_set, \
    mv_tabular_collate
from scipy.spatial.distance import cdist
'''
对于基于距离的方法，过拟合的问题通常不是主要的问题，
因为这些方法通常没有需要学习的参数，或者说参数的数量不会随着数据量的增加而增加。
然而，这并不意味着我们可以完全忽视过拟合的问题。

在基于距离的方法中，过拟合可能会以不同的方式出现。
例如，如果我们在计算距离时考虑了太多的特征，而这些特征中有些并不相关或者噪声较大，
那么我们的模型可能会过于复杂，从而导致过拟合。
为了避免这种情况，我们可以使用特征选择或者特征降维的方法来减少特征的数量。

另一种可能导致过拟合的情况是，如果我们在选择近邻时只考虑了非常少的样本，
那么我们的模型可能会过于依赖这些样本，从而忽视了数据的整体分布。
为了避免这种情况，我们可以通过增加近邻的数量来使模型更加稳健。

总的来说，虽然基于距离的方法不需要像基于模型的方法那样使用正则化来防止过拟合，
但我们仍然需要注意避免模型过于复杂或者过于依赖少数样本。
'''
def get_samples(x, y, sn, train_index, test_index, n_sample, k, if_mean=False):

    view_num = len(x) # 视图数量
    data_num = x[0].shape[0] # 样本数

    # 不缺失样本：(view_num*view_num):表示第i和j视图中都存在的所有样本的索引，可以用来作为邻居
    dismiss_view_index = [[np.array([]) for __ in range(view_num)] for _ in range(view_num)]
    # 计算每个视图中所有样本之间的欧氏距离
    print("计算距离")
    dist_all_set = [cdist(x[i], x[i], 'euclidean') for i in range(view_num)]

    # 计算每个视图中都存在的样本索引
    for i in range(view_num - 1):
        for j in range(i + 1, view_num):
            # i和j视图中都存在的样本索引，即不缺失样本
            sn_temp = sn[:, [i, j]]
            sn_temp_sum = np.sum(sn_temp, axis=1)
            sn_temp_sum[test_index] = 0
            dismiss_view_index[i][j] = dismiss_view_index[j][i] = np.where(sn_temp_sum == 2)[0]
    

    print("基于KNN填充训练集中的缺失视图")
    '''
    1. 对于没有数据或者只有一个视图的样本，使用同类样本的视图的均值填充
    2. 对于有多个视图的样本，其邻居集的构成方式为：
        对于缺失的v视图，使用另一个视图m找到K个最近邻，然后对这些邻居的v视图进行加权平均
    '''
    # step1: 获取训练集
    sn_train = sn[train_index] # sn_train：(train_num, view_num)
    x_train = [x[v][train_index] for v in range(view_num)] 
    y_train = y[train_index] 

    # step2: 筛选不完整样本，分为两类：没有数据或者只有一个视图的样本，有多个视图的样本
    inproper_indices = np.where(np.sum(sn_train, axis=1) == 0)[0]
    proper_indices = np.where(np.sum(sn_train, axis=1) > 1 and np.sum(sn_train, axis=1) < view_num)[0]

    # step3: 使用同类样本的均值填充没有数据或者只有一个视图的样本
    for i in inproper_indices:
        y_i = y_train[i][0] # 获取标签
        same_class_indices = np.where(y_train == y_i)[0] # 获取类别与当前样本相同的样本
        for v in range(view_num):
            if sn_train[i][v] == 0:
                x_train[v][i] = np.mean(x_train[v][same_class_indices], axis=0)
    
    # step4: 对有多个视图的样本，使用KNN填充
    filled_samples = []
    for i in proper_indices: # 遍历缺失视图的索引
        y_i = y_train[i][0] # 获取标签
        miss_view_index = np.nonzero(sn_train[i] == 0)[0] # 获取缺失视图的索引
        for v in miss_view_index:
            # 获取样本与其他样本的距离
            dist_v = dist_all_set[v][i]
            # 只选择类别与当前样本相同的样本
            same_class_indices = np.where(y_train == y_i)[0]
            # 根据距离计算权重
            weights = np.exp(-dist_v / np.max(dist_v))
    
    
    
    
    
    
    
    return x_train, y_train, x_test, y_test, Sn_train