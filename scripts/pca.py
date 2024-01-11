import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

from scripts.signal_dataset import SIGNAL_DATASET

# 假设你已经有了一个PyTorch Dataset实例
dataset = SIGNAL_DATASET('../dataset')  # 用你的数据集替换这里

# 从PyTorch Dataset提取数据
features = []
labels = []
for i in range(len(dataset)):
    feature, label = dataset[i]
    # if label == 'CarRoad':
    features.append(feature)  # 转换为NumPy数组
    labels.append(label)

# 将列表转换为NumPy数组
features = np.array(features)
print(features.shape)
labels = np.array(labels)

# 数据预处理：标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 应用PCA
pca = PCA(n_components=0.98)  # 示例：降维到2维
principal_components = pca.fit_transform(features_scaled)
print(principal_components.shape)

# 分析PCA结果
# 你可以检查principal_components, 或者将其转换为DataFrame进行更详细的分析
# 方差解释
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

# 主成分贡献
components = pca.components_
print("PCA Components:\n", components)