import scipy.optimize._linesearch as sc
import numpy.linalg as ln
import numpy as np
import pandas as pd
from loss import *
from learnings import *
from mse import *
from sklearn import metrics
from dataclasses import dataclass
from prettytable import PrettyTable

np.seterr(over="ignore", invalid="ignore")

def SGD(
    func: LossFunc,
    learning: Learning,
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

        a = get_a_by_learning(learning, func, batch, x, d, gradient, k, error)
        x += a * d
        k += 1
        if np.linalg.norm(gradient) ** 2 < ε or k > limit:
            break
        #print("predict, true: ", func.predict(x, 0), "MSE: ", func(x,batch))
    return x, k

MAX_ITER_RULE = 800

def test_SGD(f, learning, b_size, attemps=10):
    final_r2_score = 0
    final_k = 0
    for i in range(attemps):
        x, k = SGD(func, learning, batch_size=b_size)
        targets_predict = []
        for row in test_data_func.features:
            y_pr = x[0] + row @ x[1:]
            targets_predict.append(y_pr)
        targets = test_data_func.targets.tolist()

        final_r2_score += metrics.r2_score(targets, targets_predict)
        final_k += k
    final_r2_score /= attemps
    final_k /= attemps

    print("Number of tests: " + str(attemps))
    print("Batch size: " + str(b_size))
    print("Average r2 score: " + str(final_r2_score))
    #print("Average evaluations: " + str(final_k))


if __name__ == "__main__":
    df = pd.read_csv("data/Student_Performance.csv", header=None)
    data_train, data_test = train_test_split(df, test_size=0.25, random_state=42)
    test_data_func = LossFunc(data_test, MSE)
    func = LossFunc(data_train, MSE)

    for b_size in range(1,100,10):
        test_SGD(func,constant(0.00013), b_size)
        print("===============")

    #x, _ = SGD(func, wolfe_rule_gen(0.5, 1e-4, 0.3), batch_size=100)
    #x, _ = SGD(func, armijo_rule_gen(0.3, 0.5, 1e-3))
    #x, _ = SGD(func, dichotomy_gen(0.1, 0.8))
    #x, _ = SGD(func, constant(0.00013), batch_size=1)
    
