import scipy.optimize._linesearch as sc
import numpy.linalg as ln
import numpy as np
import pandas as pd
from nary import *
from loss import *

from dataclasses import dataclass
from prettytable import PrettyTable

np.seterr(over="ignore", invalid="ignore")

def SGD(
    func: LossFunc,
    batch_size: int = 1,
    limit: float = 1e3,
    ε: float = 1e-6,
    error: float = 0.1,
) -> Tuple[Vector, int]:
    x = np.ones(func.weights_size)
    k = 0
    while True:
        gradient = func.gradient(x, np.random.randint(0, func.rows_size, size=batch_size))
        u = -gradient
        x += u
        k += 1
        if np.linalg.norm(gradient) ** 2 < ε or k > limit:
            break
    return x, k

def MSE(
    features: Matrix,
    targets: Vector,
    weights: Vector,
    batch: Vector) -> float:
    loss = 0
    m = len(batch)
    for i in batch:
        y_predict = weights[0] + features[i] @ weights[1:]
        loss +=(targets[i] - y_predict) ** 2
    loss /= m
    return loss

if __name__ == "__main__":
    df = pd.read_csv("data/spambase.data", header=None)

    func = LossFunc(df, MSE)

    print(SGD(func))
