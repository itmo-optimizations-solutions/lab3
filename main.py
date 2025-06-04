import pandas as pd
from sklearn.model_selection import train_test_split

from loss import *
from learnings import *
from mse import *
from sklearn import metrics

np.seterr(over="ignore", invalid="ignore")

def sgd(
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
        # print("predict, true: ", func.predict(x, 0), "MSE: ", func(x,batch))
    return x, k

MAX_ITER_RULE = 800

def test_sgd(f, learning, b_size, attempts=10):
    final_r2_score = 0
    final_k = 0
    for i in range(attempts):
        x, k = sgd(func, learning, batch_size=b_size)
        targets_predict = []
        for row in test_data_func.features:
            y_pr = x[0] + row @ x[1:]
            targets_predict.append(y_pr)
        targets = test_data_func.targets.tolist()

        final_r2_score += metrics.r2_score(targets, targets_predict)
        final_k += k
    final_r2_score /= attempts
    final_k /= attempts

    print("Number of tests: " + str(attempts))
    print("Batch size: " + str(b_size))
    print("Average r2 score: " + str(final_r2_score))
    # print("Average evaluations: " + str(final_k))

if __name__ == "__main__":
    df = pd.read_csv("data/Student_Performance.csv", header=None)
    data_train, data_test = train_test_split(df, test_size=0.25, random_state=42)
    test_data_func = LossFunc(data_test, mse)
    func = LossFunc(data_train, mse)

    for b_size in range(1, 100, 10):
        test_sgd(func, constant(0.00013), b_size)

    # x, _ = sgd(func, wolfe_rule_gen(0.5, 1e-4, 0.3), batch_size=100)
    # x, _ = sgd(func, armijo_rule_gen(0.3, 0.5, 1e-3))
    # x, _ = sgd(func, dichotomy_gen(0.1, 0.8))
    # x, _ = sgd(func, constant(0.00013), batch_size=1)
