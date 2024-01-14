import pandas as pd
import numpy as np
import joblib
from ImuSignal import ImuSignal

file_path = '../dataset/BrickRoad1/001_007.csv'
df = pd.read_csv(file_path)
features_matrix = []
for col in df.columns[1:]:
    signal = ImuSignal(df[col])
    features_matrix.append(list(signal.features.values()))
features=np.array(features_matrix).flatten().reshape(1,-1)
scaler = joblib.load('../models/scaler.joblib')
features_scaled = scaler.transform(features)
model = joblib.load(f'../models/svm.joblib')
label = model.predict(features_scaled)
print(label)