from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from scripts.signal_dataset import SIGNAL_DATASET

# 假设你已经有了一个PyTorch Dataset实例
dataset = SIGNAL_DATASET('../dataset')  # 用你的数据集替换这里

# 假设 dataset 是你的PyTorch Dataset
features = []
labels = []

for i in range(len(dataset)):
    feature, label = dataset[i]
    if label == 'Bump' or label == 'SpeedBump' or label == 'BrickRoad2' or label == 'CarRoad':
        continue
    features.append(feature)  # 如果特征已经是numpy数组，则不需要转换
    labels.append(label)

# 转换为NumPy数组
features = np.array(features)
labels = np.array(labels)

# 数据预处理：标准化
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 应用PCA
pca = PCA(n_components=8)  # 例如，将数据降维到2维
features_pca = pca.fit_transform(features)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features_pca, labels, test_size=0.2, random_state=42)

# 训练KNN分类器
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 预测测试集
y_pred = knn.predict(X_test)

# 模型评估
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
