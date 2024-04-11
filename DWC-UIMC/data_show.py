# Description: This file is used to show the data in the .mat file.
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt


# 导入handwritten0.mat
data = scio.loadmat('D:/毕设/DWC-UIMC/DWC-UIMC/data/handwritten0.mat')

# 获取数据
# Assuming the data is stored in a variable called 'handwritten0'
handwritten0 = data['handwritten0']

# 展示数据
plt.imshow(handwritten0)
plt.show()