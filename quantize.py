#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


# ---- Single-layer PyTorch network (same structure as in train.py) ----
class HousingLinear(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


def quantize_to_uint8(params: np.ndarray):
    """
    Manually quantize float32 params to uint8 [0, 255] using linear mapping:
        q = round((x - min_val) / scale)
        scale = (max_val - min_val) / 255
    """
    params = params.astype(np.float32)
    min_val = float(params.min())
    max_val = float(params.max())

    if max_val == min_val:
        # Degenerate case: all params are equal
        scale = 1.0
        q = np.zeros_like(params, dtype=np.uint8)
    else:
        scale = (max_val - min_val) / 255.0
        q = np.round((params - min_val) / scale).astype(np.uint8)

    return q, scale, min_val


def main():
    os.makedirs("models", exist_ok=True)

    # Load sklearn model and test data (these must be created by running train.py first)
    data = joblib.load("models/linear_regression.joblib")
    sk_model = data["sk_model"]

    test = np.load("models/test_data.npz")
    X_test = test["X_test"].astype(np.float32)
    y_test = test["y_test"].astype(np.float32)

    # Extract parameters
    coef = sk_model.coef_.astype(np.float32).ravel()          # (n_features,)
    intercept = np.array(sk_model.intercept_, dtype=np.float32).ravel()  # (1,)

    # Save full-precision params (unquantized)
    unquant_dict = {"coef": coef, "intercept": intercept}
    joblib.dump(unquant_dict, "models/unquant_params.joblib")

    # Concatenate all params into single vector
    all_params = np.concatenate([coef, intercept])  # (n_features + 1,)

    # Quantize to uint8
    q_params, scale, min_val = quantize_to_uint8(all_params)

    # Save quantized params
    quant_dict = {
        "q_params": q_params,
        "scale": float(scale),
        "min_val": float(min_val),
        "coef_len": int(len(coef)),
    }
    joblib.dump(quant_dict, "models/quant_params.joblib")

    # Dequantize back to float32
    deq = q_params.astype(np.float32) * scale + min_val
    coef_len = quant_dict["coef_len"]
    deq_coef = deq[:coef_len]
    deq_intercept = deq[coef_len:]

    # Build PyTorch model with dequantized weights
    in_features = coef.shape[0]
    model_quant = HousingLinear(in_features)

    with torch.no_grad():
        w = torch.from_numpy(deq_coef.reshape(1, -1))   # (1, in_features)
        b = torch.from_numpy(deq_intercept.reshape(1,)) # (1,)
        model_quant.linear.weight.copy_(w)
        model_quant.linear.bias.copy_(b)

    # Compare R^2: original sklearn vs quantized model
    y_pred_sk = sk_model.predict(X_test)
    r2_sk = r2_score(y_test, y_pred_sk)

    model_quant.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test)
        y_pred_quant = model_quant(X_tensor).numpy()
    r2_quant = r2_score(y_test, y_pred_quant)

    # Size comparison
    size_unquant = os.path.getsize("models/unquant_params.joblib") / 1024.0
    size_quant = os.path.getsize("models/quant_params.joblib") / 1024.0

    print("\n=== Quantization Report ===")
    print(f"Original sklearn R^2        : {r2_sk:.6f}")
    print(f"Quantized model R^2         : {r2_quant:.6f}")
    print(f"unquant_params.joblib size  : {size_unquant:.2f} KB")
    print(f"quant_params.joblib size    : {size_quant:.2f} KB")
    print("============================")


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




