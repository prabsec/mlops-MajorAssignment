import os
import joblib
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def main():
    # 1) Load dataset
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # 2) Split into train/test (same as in train.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Load the saved model (created by train.py in CI before building the image)
    model_path = os.path.join("models", "housing_linear.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)

    # 4) Run prediction and compute R^2
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"[PREDICT] R^2 score on test set: {r2:.4f}")


if __name__ == "__main__":
    main()
