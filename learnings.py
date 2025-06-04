import numpy as np
from typing import Callable
from loss import LossFunc, Vector

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
