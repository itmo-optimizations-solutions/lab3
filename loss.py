import numpy as np
from typing import Callable, Tuple

from pandas import DataFrame
from sklearn.model_selection import train_test_split

Vector = np.ndarray
Matrix = np.ndarray
Loss = Callable[[Matrix, Vector, Vector, Vector], float]
SYSTEM_EPS = 1e-6

class LossFunc:
    features: Matrix
    targets: Vector
    weights_size: int
    rows_size: int
    loss_eval: Loss

    def __init__(self,
                 df: DataFrame,
                 loss_eval: Loss,
                 ) -> None:
        self.features = df.iloc[:, :-1].to_numpy()
        self.targets = df.iloc[:, -1].to_numpy()
        self.rows_size = self.features.shape[0]
        self.weights_size = self.features.shape[1] + 1
        self.loss_eval = loss_eval

    def __call__(self,
                 weights: Vector,
                 batch: Vector
                 ) -> float:
        return self.loss_eval(self.features, self.targets, weights, batch)

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

    def predict(self, weights:Vector, test_index: int) -> Tuple[int, int]:
        y_predict = weights[0] + self.features[test_index] @ weights[1:]
        y = self.targets[test_index]
        return y_predict, y
