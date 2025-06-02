import numpy as np
from typing import Callable, Tuple

Vector = np.ndarray
# SYSTEM_EPS = np.sqrt(np.finfo(float).eps) # 1.5e-8
SYSTEM_EPS = 1e-6

class NaryFunc:
    Type = Callable[[list[Vector]], float]
    func: Type
    g_count: int
    h_count: int

    def __init__(self, func: Type) -> None:
        self.func = func
        self.g_count = 0
        self.h_count = 0

    def __call__(self, x: Vector) -> float:
        return self.func(*x)

    def gradient(self, x: Vector, ε: float = SYSTEM_EPS) -> Vector:
        self.g_count += 1
        gradient = np.zeros_like(x)
        size = len(x)
        for i in range(size):
            dx = np.zeros(size)
            h = max(ε, ε * abs(x[i]))
            dx[i] = h
            gradient[i] = (self(x + dx) - self(x - dx)) / (2 * h)
        return gradient

    def hessian(self, x: Vector, ε: float = SYSTEM_EPS) -> Vector:
        self.h_count += 1
        size = len(x)
        hessian = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                dx = np.zeros(size)
                dy = np.zeros(size)

                hx = max(ε, ε * abs(x[i]))
                hy = max(ε, ε * abs(x[j]))

                dx[i] = hx
                dy[j] = hy

                hessian[i, j] = (self(x + dx + dy) - self(x + dx - dy) -
                                 self(x - dx + dy) + self(x - dx - dy)) / (4 * hx * hy)

        return hessian

Scheduling = Callable[[int], float]
Rule = Callable[[NaryFunc, Vector, Vector], float]
Condition = Callable[[NaryFunc, Vector, Vector, Vector], float]

Learning = Scheduling | Rule | Condition
Descent = Callable[[NaryFunc, Vector, Learning], Tuple[Vector, int, int, int, list]]
