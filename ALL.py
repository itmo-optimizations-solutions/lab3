import time
import psutil
import os
import numpy as np

from typing import Callable, Tuple
from pandas import DataFrame

Vector = np.ndarray
Matrix = np.ndarray
Loss = Callable[[Matrix, Vector, Vector, Vector], float]

SYSTEM_EPS = 1e-6

class PerformanceCounter:
    additions: int
    multiplications: int
    divisions: int
    comparisons: int
    memory_peak: int
    start_time: float
    end_time: float

    def __init__(self):
        self.reset()

    def reset(self):
        self.additions = 0
        self.multiplications = 0
        self.divisions = 0
        self.comparisons = 0
        self.memory_peak = 0
        self.start_time = None
        self.end_time = None

    def start_timing(self):
        self.start_time = time.time()
        self.memory_peak = self.get_memory_usage()

    def end_timing(self):
        self.end_time = time.time()
        current_memory = self.get_memory_usage()
        if current_memory > self.memory_peak:
            self.memory_peak = current_memory

    @staticmethod
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def get_execution_time(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    def add_operations(self, adds=0, mults=0, divs=0, comps=0):
        self.additions += adds
        self.multiplications += mults
        self.divisions += divs
        self.comparisons += comps

    def get_total_operations(self):
        return self.additions + self.multiplications + self.divisions + self.comparisons

    def get_report(self):
        return {
            'total_ops': self.get_total_operations(),
            'additions': self.additions,
            'multiplications': self.multiplications,
            'divisions': self.divisions,
            'comparisons': self.comparisons,
            'memory_mb': self.memory_peak,
            'time_sec': self.get_execution_time()
        }

perf_counter = PerformanceCounter()

class LossFunc:
    features: Matrix
    targets: Vector
    weights_size: int
    rows_size: int
    loss_eval: Loss

    def __init__(self, df: DataFrame, loss_eval: Loss) -> None:
        self.features = df.iloc[:, :-1].to_numpy()
        self.targets = df.iloc[:, -1].to_numpy()
        self.rows_size = self.features.shape[0]
        self.weights_size = self.features.shape[1] + 1
        self.loss_eval = loss_eval

    def __call__(self, weights: Vector, batch: Vector) -> float:
        return self.loss_eval(self.features, self.targets, weights, batch)

    def gradient(self, weights: Vector, batch: Vector, ε: float = SYSTEM_EPS) -> Vector:
        gradient = np.zeros_like(weights)
        size = len(weights)

        for i in range(size):
            dx = np.zeros(size)
            h = ε * max(1, abs(weights[i]))
            dx[i] = h

            loss_plus = self(weights + dx, batch)
            loss_minus = self(weights - dx, batch)
            gradient[i] = (loss_plus - loss_minus) / (2 * h)

            perf_counter.add_operations(adds=size * 2, mults=2, divs=1)

        return gradient

    def predict(self, weights: Vector, test_index: int) -> Tuple[int, int]:
        y_predict = weights[0] + self.features[test_index] @ weights[1:]
        y = self.targets[test_index]

        perf_counter.add_operations(adds=1, mults=len(weights) - 1)

        return y_predict, y

Scheduling = Callable[[int], float]
Rule = Callable[[LossFunc, Vector, Vector, Vector], float]
Condition = Callable[[LossFunc, Vector, Vector, Vector, Vector], float]

Learning = Scheduling | Rule | Condition

def is_scheduling(algorithm) -> bool:
    return hasattr(algorithm, "__code__") and algorithm.__code__.co_argcount == 1

def is_rule(algorithm) -> bool:
    return hasattr(algorithm, "__code__") and algorithm.__code__.co_argcount == 4

def is_condition(algorithm) -> bool:
    return hasattr(algorithm, "__code__") and algorithm.__code__.co_argcount == 5

def get_a_by_learning(
    learning: Learning,
    func: LossFunc,
    batch: Vector,
    x: Vector,
    d: Vector,
    gradient: Vector,
    k: int,
    error: float
) -> float:
    if is_scheduling(learning):
        return learning(k)
    elif is_rule(learning):
        return learning(func, batch, x, d)
    elif is_condition(learning):
        return learning(func, batch, x, d, gradient)
    else:
        return error

def _h(k: int) -> float:
    return 1 / (k + 1) ** 0.5

def constant(λ: float) -> Scheduling:
    return lambda k: λ

def geometric() -> Scheduling:
    return lambda k: _h(k) / 2 ** k

def exponential_decay(λ: float) -> Scheduling:
    return lambda k: _h(k) * np.exp(-λ * k)

def polynomial_decay(α: float, β: float) -> Scheduling:
    return lambda k: _h(k) * (β * k + 1) ** -α

def armijo_rule_cond(α: float, q: float, c: float) -> Condition:
    return lambda func, batch, x, direction, gradient: (
        armijo_rule(func, batch, x, direction, gradient, α=α, q=q, c=c))

def wolfe_rule_cond(α: float, c1: float, c2: float) -> Condition:
    return lambda func, batch, x, direction, _: (
        wolfe_rule(func, batch, x, direction, α=α, c1=c1, c2=c2))

def dichotomy_gen(a: float, b: float, eps: float = 1e-6) -> Rule:
    return lambda func, batch, x, direction: (
        dichotomy(func, batch, x, direction, a=a, b=b, eps=eps))

MAX_ITER_RULE = 800

def armijo_rule(
    func: LossFunc,
    batch: Vector,
    x: Vector,
    direction: Vector,
    gradient: Vector,
    α: float,
    q: float,
    c: float,
) -> float | None:
    for _ in range(MAX_ITER_RULE):
        tmp = func(x, batch) + c * α * np.dot(gradient, direction)
        if func(x + α * direction, batch) <= tmp:
            return α
        α *= q
    return None

def wolfe_rule(
    func: LossFunc,
    batch: Vector,
    x: Vector,
    direction: Vector,
    α: float,
    c1: float,
    c2: float,
) -> float | None:
    for _ in range(MAX_ITER_RULE):
        tmp = func(x, batch) + c1 * α * np.dot(-direction, direction)
        if func(x + α * direction, batch) > tmp:
            α *= 0.5
        elif np.dot(
            func.gradient(x + α * direction, batch),
            direction
        ) < c2 * np.dot(-direction, direction):
            α *= 1.5
        else:
            return α
    return None

def dichotomy(
    func: LossFunc,
    batch: Vector,
    x: Vector,
    direction: Vector,
    a: float,
    b: float,
    eps: float
) -> float:
    def phi(alpha: float) -> float:
        return func(x + alpha * direction, batch)

    while (b - a) > eps:
        c = (a + b) / 2
        f_c = phi(c)
        a1 = (a + c) / 2.0
        f_a1 = phi(a1)
        b1 = (c + b) / 2.0
        f_b1 = phi(b1)
        if f_a1 < f_c:
            b = c
        elif f_c > f_b1:
            a = c
        else:
            a, b = a1, b1
    return (a + b) / 2.0

def mse(
    features: Matrix,
    targets: Vector,
    weights: Vector,
    batch: Vector
) -> float:
    loss = 0
    m = len(batch)

    for i in batch:
        y_predict = weights[0] + features[i] @ weights[1:]
        loss += (targets[i] - y_predict) ** 2

        perf_counter.add_operations(
            adds=2,
            mults=len(weights) - 1 + 1,
        )

    loss /= m
    perf_counter.add_operations(divs=1)

    return loss

def mse_l1(λ: float):
    def loss_func(features: Matrix, targets: Vector, weights: Vector, batch: Vector) -> float:
        mse_loss = mse(features, targets, weights, batch)
        l1_penalty = λ * np.sum(np.abs(weights[1:]))

        perf_counter.add_operations(
            adds=len(weights) - 1 + 1,
            mults=1
        )

        return mse_loss + l1_penalty

    return loss_func

def mse_l2(λ: float):
    def loss_func(features: Matrix, targets: Vector, weights: Vector, batch: Vector) -> float:
        mse_loss = mse(features, targets, weights, batch)
        l2_penalty = λ * np.sum(weights[1:] ** 2)

        perf_counter.add_operations(
            adds=len(weights) - 1 + 1,
            mults=len(weights) - 1 + 1
        )

        return mse_loss + l2_penalty

    return loss_func

def mse_elastic_net(λ1: float, λ2: float):
    def loss_func(features: Matrix, targets: Vector, weights: Vector, batch: Vector) -> float:
        mse_loss = mse(features, targets, weights, batch)
        l1_penalty = λ1 * np.sum(np.abs(weights[1:]))
        l2_penalty = λ2 * np.sum(weights[1:] ** 2)

        perf_counter.add_operations(
            adds=(len(weights) - 1) * 2 + 2,
            mults=(len(weights) - 1) + 2
        )

        return mse_loss + l1_penalty + l2_penalty

    return loss_func

np.seterr(over="ignore", invalid="ignore")

def sgd_with_performance(
    func: LossFunc,
    learning: Learning,
    batch_size: int = 10,
    limit: float = 1e3,
    ε: float = 1e-6,
    error: float = 0.1,
) -> Tuple[Vector, int, dict]:
    perf_counter.reset()
    perf_counter.start_timing()

    x = np.random.rand(func.weights_size)
    k = 0

    while True:
        current_memory = perf_counter.get_memory_usage()
        if current_memory > perf_counter.memory_peak:
            perf_counter.memory_peak = current_memory

        batch = np.random.choice(func.rows_size, size=batch_size, replace=False)
        gradient = func.gradient(x, batch)
        d = -gradient

        a = get_a_by_learning(learning, func, batch, x, d, gradient, k, error)
        x += a * d

        perf_counter.add_operations(adds=len(x), mults=len(x))
        k += 1

        norm_grad = np.linalg.norm(gradient) ** 2
        perf_counter.add_operations(mults=len(gradient) + 1, comps=2)

        if norm_grad < ε or k > limit:
            break

    perf_counter.end_timing()
    performance_report = perf_counter.get_report()
    return x, k, performance_report

def sgd_momentum_with_performance(
    func: LossFunc,
    learning: Learning,
    momentum: float = 0.9,
    batch_size: int = 10,
    limit: float = 1e3,
    ε: float = 1e-6,
    error: float = 0.1,
) -> Tuple[Vector, int, dict]:
    perf_counter.reset()
    perf_counter.start_timing()

    x = np.random.rand(func.weights_size)
    velocity = np.zeros_like(x)
    k = 0

    while True:
        current_memory = perf_counter.get_memory_usage()
        if current_memory > perf_counter.memory_peak:
            perf_counter.memory_peak = current_memory

        batch = np.random.choice(func.rows_size, size=batch_size, replace=False)
        gradient = func.gradient(x, batch)

        a = get_a_by_learning(learning, func, batch, x, -gradient, gradient, k, error)
        velocity = momentum * velocity - a * gradient
        x += velocity

        perf_counter.add_operations(
            adds=len(x) * 2,
            mults=len(x) * 2
        )
        k += 1

        norm_grad = np.linalg.norm(gradient) ** 2
        perf_counter.add_operations(mults=len(gradient) + 1, comps=2)

        if norm_grad < ε or k > limit:
            break

    perf_counter.end_timing()
    performance_report = perf_counter.get_report()
    return x, k, performance_report
