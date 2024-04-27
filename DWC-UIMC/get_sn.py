import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
import random


def get_sn(view_num, alldata_len, missing_rate):
    """随机生成不完整的数据信息，使用完整视图数据模拟缺失视图数据
    :param view_num:视图数量
    :param alldata_len:样本数量
    :param missing_rate:缺失率，Defined in section 3.2 of the paper
    :return:Sn——缺失数据信息
    """
    one_rate = 1-missing_rate # 完整视图数据比例
    if one_rate <= (1 / view_num): 
        # 如果完整视图数据比例小于等于1/视图数量，意味着每个视图中的完整数据比例都不足以构成一个完整的数据集。
        # 每个样本只有一个视图的数据是完整的，其他视图的数据是缺失的。
        '''
        假设我们有一个3视图，5样本的数据集，获得随机整数矩阵为：
        [[0],
        [2],
        [1],
        [2],
        [0]]
        转换为独热编码后：
        [[1, 0, 0],
         [0, 0, 1],
         [0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]]
        代表第一个样本的完整视图数据是第0个视图，第二个样本的完整视图数据是第2个视图，以此类推。
        '''
        enc = OneHotEncoder() # OneHotEncoder()用于将整数标签转换为独热编码

        # 生成一个大小为alldata_len*1的随机整数矩阵，元素取值范围为[0,view_num)
        # 然后将其转换为独热编码（alldata_len, view_num），代表每个样本的完整视图数据是哪一个视图
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1 # 定义一个误差值
    if one_rate == 1: 
        # 如果完整视图数据比例为1，即没有缺失数据
        # 随机生成一个完整的数据信息（全1矩阵）
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    # 当完整视图数据比例大于1/视图数量时，随机生成缺失数据信息
    while error >= 0.005: # 当误差大于0.005时，重新生成缺失数据信息
        '''
        每次循环进行一次随机生成缺失数据信息的过程，直到生成的缺失数据信息满足完整视图数据比例的要求。
        包括以下几个步骤：
        1. 在每个样本中随机保留一个视图的基础上，计算仍需保留的完整数据的数量one_num及完整数据比例ratio
        2. 随机生成保留矩阵，决定哪些视图的数据是完整的
        3. 计算仍需保留的完整数据的数量one_num_iter及完整数据比例ratio
        4. 再次随机生成保留矩阵，进行第二次随机保留操作
        5. 与所需完整数据比例one_rate进行比较，计算误差error
        '''
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()

        # 因为在这个模型中，每个样本至少有一个视图是完整的
        # 打算保留数量（具体到某个视图的某个样本）=视图数量*样本数量*完整率-样本数量
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len) # 完整数据比例

        # 生成一个随机矩阵matrix_iter，元素是0或1，取决于随机生成的数是否小于ratio * 100
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
        # 计算matrix_iter和view_preserve的和大于1的元素数量，即当前已经保留的完整数据的数量
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int_))

        # 重新计算完整数据的数量one_num_iter，即还应该保留的数量=打算保留数量/(1-已经保留的完整数据的数量/打算保留数量)
        one_num_iter = one_num / (1 - a / one_num)
        # 在实际的数据处理中，可能会有一些数据被多次保留，导致实际保留的数量 a 大于理论上应该保留的数量
        # 所以如果直接使用 one_num - a，可能会得到一个负数，这是不合理的。

        # 重新计算完整数据比例
        ratio = one_num_iter / (view_num * alldata_len)

        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int_)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio) # 误差=完整数据比例-实际完整数据比例
    return matrix

def save_Sn(Sn,path):
    np.savetxt(path, Sn, delimiter=',', fmt='%d')


def load_Sn(str_name):
    return np.loadtxt(str_name + '.csv', delimiter=',')

if __name__ == '__main__':
    missrate = 0.2
    dataset_name_list = ['handwritten0.mat','BRCA.mat','ROSMAP.mat']
    # dataset_name = dataset_name_list[0]
    # dataset_name = dataset_name_list[1]
    dataset_name = dataset_name_list[2]

    if dataset_name == 'handwritten0.mat':
        sn = get_sn(6,2000,missrate) # handwrite0.mat
        save_Sn(sn,'./handwritten_sh_'+ str(missrate)[2:]+'.csv')
    elif dataset_name == 'BRCA.mat':
        sn = get_sn(3,875,missrate) # BRCA.mat
        save_Sn(sn,'./BRCA_sh_'+ str(missrate)[2:]+'.csv')
    elif dataset_name == 'ROSMAP.mat':
        sn = get_sn(3,351,missrate) # ROSMAP.mat
        save_Sn(sn,'./ROSMAP_sh_'+ str(missrate)[2:]+'.csv')


