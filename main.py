import scipy.optimize._linesearch as sc
import numpy.linalg as ln
import numpy as np
import pandas as pd
from loss import *
from sklearn import metrics
from dataclasses import dataclass
from prettytable import PrettyTable

np.seterr(over="ignore", invalid="ignore")

def SGD(
    func: LossFunc,
    batch_size: int = 10,
    limit: float = 1e3,
    ε: float = 1e-6,
    error: float = 0.1,
) -> Tuple[Vector, int]:
    x = np.random.rand(func.weights_size)
    k = 0
    while True:
        batch = np.random.choice(func.rows_size, size=batch_size, replace=False)
        gradient = func.gradient(x, batch)
        d = -gradient
        a = wolfe_rule(func, batch, x, d, α=0.5, c1=1e-4, c2=0.3)
        a = error if a is None else a
        x += a * d
        k += 1
        if np.linalg.norm(gradient) ** 2 < ε or k > limit:
            break
        print("predict, true: ", func.predict(x, 0), "MSE: ", func(x,batch))
    return x, k

def wolfe_rule(
    func: LossFunc,
    batch : Vector,
    x: Vector,
    direction: Vector,
    α: float,
    c1: float,
    c2: float,
) -> float | None:
    for _ in range(MAX_ITER_RULE):
        if func(x + α * direction,batch) > func(x,batch) + c1 * α * np.dot(-direction, direction):
            α *= 0.5
        elif np.dot(func.gradient(x + α * direction,batch), direction) < c2 * np.dot(-direction, direction):
            α *= 1.5
        else:
            return α
    return None

MAX_ITER_RULE = 800

if __name__ == "__main__":
    df = pd.read_csv("data/Student_Performance.csv", header=None)
    data_train, data_test = train_test_split(df, test_size=0.25, random_state=42)
    test_data_func = LossFunc(data_test)
    func = LossFunc(data_train)
    x, _ = SGD(func)
    targets_predict = []
    for row in test_data_func.features:
        y_pr = x[0] + row @ x[1:]
        targets_predict.append(y_pr)
    targets = test_data_func.targets.tolist()
    print("r2 score:", metrics.r2_score(targets, targets_predict))
