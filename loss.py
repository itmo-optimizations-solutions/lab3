import numpy as np
from typing import Callable, Tuple

from pandas import DataFrame

Vector = np.ndarray
Matrix = np.ndarray
SYSTEM_EPS = 1e-6

class LossFunc:
    features : Matrix
    target : Vector
    weights_size : int
    rows_size : int

    def __init__(self,
                 df : DataFrame,
                 loss_fn: Callable[[Matrix, Vector, Vector, Vector], float]
                 ) -> None:
        self.features = df.iloc[:, :-1].to_numpy()
        self.target = df.iloc[:, -1].to_numpy()
        self.loss_fn = loss_fn
        self.rows_size =  self.features.shape[0]
        self.weights_size = self.features.shape[1] + 1

    def __call__(self, weights: Vector, batch : Vector) -> float:
        return self.loss_fn(self.features, self.target, weights, batch)

    def gradient(self, weights: Vector, batch : Vector, ε: float = SYSTEM_EPS) -> Vector:
        gradient = np.zeros_like(weights)
        size = len(weights)
        for i in range(size):
            dx = np.zeros(size)
            h = max(ε, ε * abs(weights[i]))
            dx[i] = h
            gradient[i] = (self(weights + dx, batch) - self(weights - dx, batch)) / (2 * h)
        return gradient