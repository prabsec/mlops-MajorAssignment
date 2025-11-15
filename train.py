#!/usr/bin/env python
# coding: utf-8

# In[1]:


# train.py
import os  # to work with the file system
import joblib  # to save and load python objects
import numpy as np  # Numerical arrays

from sklearn.datasets import fetch_california_housing  # features of housing dataset and their prices
from sklearn.model_selection import train_test_split  # randomly split data into train and test
from sklearn.linear_model import LinearRegression  # Linear regression
from sklearn.metrics import r2_score  # performance metric

import torch
import torch.nn as nn


# Single-layer PyTorch network: y = xW^T + b
class HousingLinear(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


def main():
    # Load dataset
    data = fetch_california_housing()
    X = data.data.astype(np.float32)      # shape: (N, 8) Each row = one house, each column = one feature
    y = data.target.reshape(-1, 1).astype(np.float32)

    # Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train sklearn LinearRegression
    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = sk_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"[train.py] Sklearn LinearRegression test R^2: {r2:.4f}")

    # Create output directory
    os.makedirs("models", exist_ok=True)

    # Save sklearn model and metadata via joblib
    joblib.dump(
        {
            "sk_model": sk_model,
            "feature_names": data.feature_names,
            "r2_test": r2,
            "split_params": {
                "test_size": 0.2,
                "random_state": 42,
            },
        },
        "models/linear_regression.joblib",
    )

    # Save test set for later use (prediction / quantization)
    np.savez("models/test_data.npz", X_test=X_test, y_test=y_test)

    print("[train.py] Saved models/linear_regression.joblib and models/test_data.npz")

    # --------- NEW PART: create single-layer PyTorch NN from sklearn weights ---------

    in_features = X.shape[1]  # 8 features
    torch_model = HousingLinear(in_features)

    # sklearn.coef_: shape (1, n_features) or (n_features,)
    coef = sk_model.coef_.astype(np.float32)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)  # (1, n_features)

    # sklearn.intercept_: shape (1,) or scalar
    intercept = np.array(sk_model.intercept_, dtype=np.float32).reshape(1,)

    # Copy weights into torch model
    with torch.no_grad():
        # torch linear: weight shape = (1, in_features), bias shape = (1,)
        torch_model.linear.weight.copy_(torch.from_numpy(coef))
        torch_model.linear.bias.copy_(torch.from_numpy(intercept))

    # Optional: verify R^2 from torch model on the same test set
    torch_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test)  # (N, 8)
        y_pred_torch = torch_model(X_test_tensor).numpy()  # (N, 1)

    r2_torch = r2_score(y_test, y_pred_torch)
    print(f"[train.py] PyTorch model test R^2 (copied weights): {r2_torch:.4f}")

    # Save PyTorch model checkpoint
    torch.save(
        {
            "state_dict": torch_model.state_dict(),
            "in_features": in_features,
        },
        "models/torch_model.pth",
    )
    print("[train.py] Saved models/torch_model.pth")


if __name__ == "__main__":
    main()


# In[15]:


# train.py
import os  # to work with the file system
import joblib  # to save and load python objects
import numpy as np  # Numerical arrays

from sklearn.datasets import fetch_california_housing  # features of housing dataset and their prices
from sklearn.model_selection import train_test_split  # randomly split data into train and test
from sklearn.linear_model import LinearRegression  # Linear regression
from sklearn.metrics import r2_score  # performance metric

import torch
import torch.nn as nn


# Single-layer PyTorch network: y = xW^T + b
class HousingLinear(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


def main():
    # Load dataset
    data = fetch_california_housing()
    X = data.data.astype(np.float32)      # shape: (N, 8) Each row = one house, each column = one feature
    y = data.target.reshape(-1, 1).astype(np.float32)

    # Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train sklearn LinearRegression
    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = sk_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"[train.py] Sklearn LinearRegression test R^2: {r2:.4f}")

    # Create output directory
    os.makedirs("models", exist_ok=True)

    # Save sklearn model and metadata via joblib
    joblib.dump(
        {
            "sk_model": sk_model,
            "feature_names": data.feature_names,
            "r2_test": r2,
            "split_params": {
                "test_size": 0.2,
                "random_state": 42,
            },
        },
        "models/linear_regression.joblib",
    )

    # Save test set for later use (prediction / quantization)
    np.savez("models/test_data.npz", X_test=X_test, y_test=y_test)

    print("[train.py] Saved models/linear_regression.joblib and models/test_data.npz")

    # --------- NEW PART: create single-layer PyTorch NN from sklearn weights ---------

    in_features = X.shape[1]  # 8 features
    torch_model = HousingLinear(in_features)

    # sklearn.coef_: shape (1, n_features) or (n_features,)
    coef = sk_model.coef_.astype(np.float32)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)  # (1, n_features)

    # sklearn.intercept_: shape (1,) or scalar
    intercept = np.array(sk_model.intercept_, dtype=np.float32).reshape(1,)

    # Copy weights into torch model
    with torch.no_grad():
        # torch linear: weight shape = (1, in_features), bias shape = (1,)
        torch_model.linear.weight.copy_(torch.from_numpy(coef))
        torch_model.linear.bias.copy_(torch.from_numpy(intercept))

    # Optional: verify R^2 from torch model on the same test set
    torch_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test)  # (N, 8)
        y_pred_torch = torch_model(X_test_tensor).numpy()  # (N, 1)

    r2_torch = r2_score(y_test, y_pred_torch)
    print(f"[train.py] PyTorch model test R^2 (copied weights): {r2_torch:.4f}")

    # Save PyTorch model checkpoint
    torch.save(
        {
            "state_dict": torch_model.state_dict(),
            "in_features": in_features,
        },
        "models/torch_model.pth",
    )
    print("[train.py] Saved models/torch_model.pth")


if __name__ == "__main__":
    main()


# In[ ]:




