import numpy as np

# 创建原始数据点
original_data = np.random.rand(176)

# 创建插值点的索引
interpolation_indices = np.linspace(0, 175, 112, dtype=int)

# 使用线性插值来获取插值点的值
interpolated_data = np.interp(interpolation_indices, np.arange(176), original_data)

# interpolated_data 现在包含了平均取出的112个数据点
print(interpolation_indices)
print(len(interpolated_data))