import numpy as np
from typing import Callable, Tuple

from pandas import DataFrame
from sklearn.model_selection import train_test_split

Vector = np.ndarray
Matrix = np.ndarray
SYSTEM_EPS = 1e-6

class LossFunc:
    features: Matrix
    targets: Vector
    weights_size: int
    rows_size: int

    def __init__(self,
                 df: DataFrame,
                 ) -> None:
        self.features = df.iloc[:, :-1].to_numpy()
        self.targets = df.iloc[:, -1].to_numpy()
        self.rows_size = self.features.shape[0]
        self.weights_size = self.features.shape[1] + 1

    def __call__(self,
                 weights: Vector,
                 batch: Vector
                 ) -> float:
        return self.MSE(weights, batch)

    def gradient(self,
                 weights: Vector,
                 batch: Vector,
                 ε: float = SYSTEM_EPS
                 ) -> Vector:
        gradient = np.zeros_like(weights)
        size = len(weights)
        for i in range(size):
            dx = np.zeros(size)
            h = ε * max(1, abs(weights[i]))
            dx[i] = h
            gradient[i] = (self(weights + dx, batch) - self(weights - dx, batch)) / (2 * h)
        return gradient

    def MSE(self,
            weights: Vector,
            batch: Vector
            ) -> float:
        loss = 0
        m = len(batch)
        for i in batch:
            y_predict = weights[0] + self.features[i] @ weights[1:]
            loss += (self.targets[i] - y_predict) ** 2
        loss /= m
        return loss

    def L2(self, weights, batch, lambda_reg=0.01) -> float:
        loss = 0.0
        for i in batch:
            y_pred = weights[0] + self.features[i] @ weights[1:]
            loss += (self.targets[i] - y_pred) ** 2
        loss = loss / len(batch) + lambda_reg * np.sum(weights ** 2)  # L2-регуляризация
        return loss

    def predict(self, weights:Vector, test_index: int) -> Tuple[int, int]:
        y_predict = weights[0] + self.features[test_index] @ weights[1:]
        y = self.targets[test_index]
        return y_predict, y
