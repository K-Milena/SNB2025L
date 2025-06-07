import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data  # input features: 150 samples, 4 features
y = iris.target  # target classes (0: setosa, 1: versicolor, 2: virginica)
labels = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# Normalize data to [0, 1]
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Print shapes to confirm
print("Input shape:", X.shape)
print("Normalized shape:", X_normalized.shape)
print("Classes:", np.unique(y))
