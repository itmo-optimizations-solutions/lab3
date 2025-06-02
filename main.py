import scipy.optimize._linesearch as sc
import numpy.linalg as ln
import numpy as np
from nary import *

from dataclasses import dataclass
from prettytable import PrettyTable

np.seterr(over="ignore", invalid="ignore")

def is_scheduling(algorithm) -> bool:
    return hasattr(algorithm, "__code__") and algorithm.__code__.co_argcount == 1

def is_condition(algorithm) -> bool:
    return hasattr(algorithm, "__code__") and algorithm.__code__.co_argcount == 4

def get_a_by_learning(
    learning: Learning,
    func: NaryFunc,
    x: Vector,
    d: Vector,
    gradient: Vector,
    k: int,
    error: float
) -> float:
    if is_scheduling(learning):
        α = learning(k)
    elif is_condition(learning):
        α = learning(func, x, d, gradient)
    else:
        α = learning(func, x, d)
    return error if α is None else α


def find_direction(func, gradient, x, ε):
    radius_area = ε ** 0.5
    func_val = func(x)
    for att in range(20):
        random_multipliers = np.random.randint(-10, 11, x.shape) / 10.0
        eps_array = radius_area * random_multipliers
        if func(x + eps_array) < func_val:
            x += eps_array
            max_state = True
            gradient = func.gradient(x)
            break
    return gradient

def SGD(
    func: LossFunc,
    start: Vector,
    learning: Learning,
    regular: Regularization,
    batch_size: int = 1,
    limit: float = 1e3,
    ε: float = 1e-6,
    error: float = 0.1,
) -> Tuple[Vector, int, int, list, int]:
    x = start.copy()
    trajectory = [x.copy()]
    k = 0

    while True:
        gradient = func.gradient(x)
        max_state = False

        if np.allclose(gradient, 0):
            gradient = find_direction(func, gradient, x, ε)

        u = -gradient
        alpha = get_a_by_learning(learning, func, x, u, gradient, k, error)
        x += alpha * u
        trajectory.append(x.copy())

        k += 1
        if not max_state and np.linalg.norm(gradient) ** 2 < ε or k > limit:
            break

    grad_count = func.g_count
    func.g_count = 0
    return x, grad_count, func.h_count, k, trajectory 

# === Learnings

def h(k: int) -> float:
    return 1 / (k + 1) ** 0.5

def constant(λ: float) -> Scheduling:
    return lambda k: λ

def geometric() -> Scheduling:
    return lambda k: h(k) / 2 ** k

def exponential_decay(λ: float) -> Scheduling:
    return lambda k: h(k) * np.exp(-λ * k)

def polynomial_decay(α: float, β: float) -> Scheduling:
    return lambda k: h(k) * (β * k + 1) ** -α

MAX_ITER_RULE = 80

def armijo_rule(
    func: NaryFunc,
    x: Vector,
    direction: Vector,
    gradient: Vector,
    α: float,
    q: float,
    c: float,
) -> float | None:
    for _ in range(MAX_ITER_RULE):
        if func(x + α * direction) <= func(x) + c * α * np.dot(gradient, direction):
            return α
        α *= q
    return None

def wolfe_rule(
    func: NaryFunc,
    x: Vector,
    direction: Vector,
    gradient: Vector,
    α: float,
    c1: float,
    c2: float,
) -> float | None:
    for _ in range(MAX_ITER_RULE):
        if func(x + α * direction) > func(x) + c1 * α * np.dot(gradient, direction):
            α *= 0.5
        elif np.dot(func.gradient(x + α * direction), direction) < c2 * np.dot(gradient, direction):
            α *= 1.5
        else:
            return α
    return None

def armijo_rule_gen(α: float, q: float, c: float) -> Condition:
    return lambda func, x, direction, gradient: armijo_rule(func, x, direction, gradient, α=α, q=q, c=c)

def wolfe_rule_gen(α: float, c1: float, c2: float) -> Condition:
    return lambda func, x, direction, gradient: wolfe_rule(func, x, direction, gradient, α=α, c1=c1, c2=c2)

def scipy_wolfe(func: NaryFunc, x: Vector, direction: Vector, gradient: Vector) -> float:
    return sc.line_search_wolfe1(func, func.gradient, x, direction)[0]

def scipy_armijo(func: NaryFunc, x: Vector, direction: Vector, gradient: Vector) -> float:
    return sc.scalar_search_armijo(
        phi=lambda α: func(x + α * direction),
        phi0=func(x),
        derphi0=np.dot(gradient, direction)
    )[0]

def dichotomy_gen(a: float, b: float, eps: float = 1e-6) -> Rule:
    return lambda func, x, direction: dichotomy(func, x, direction, a=a, b=b, eps=eps)

def dichotomy(
    func: NaryFunc,
    x: np.ndarray,
    direction: np.ndarray,
    a: float,
    b: float,
    eps: float
) -> float:
    def phi(alpha: float) -> float:
        return func(x + alpha * direction)

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

# === Launcher

@dataclass
class Algorithm:
    name: str
    meta: str
    algorithm: Learning

    def get_data(self, func: NaryFunc, start: Vector, descent: Descent) -> list:
        x, grad_count, hes_count, k, _ = descent(func, start, self.algorithm)
        return [self.name] + [self.meta] + list(x) + [grad_count] + [hes_count] + [k]

@dataclass
class SciAlgorithm:
    name: str
    meta: str
    evaluator: Callable[[NaryFunc, Vector], Tuple[Vector, int, int, int]]

    def get_data(self, func: NaryFunc, start: Vector, _: Descent) -> list:
        x, grad_count, h_count, k = self.evaluator(func, start)
        func.g_count = 0
        func.h_count = 0
        return [self.name] + [self.meta] + list(x) + [grad_count] + [h_count] + [k]

KNOWN = [
    Algorithm("Constant", "λ=0.3", constant(λ=0.3)),
    Algorithm("Constant", "λ=0.003", constant(λ=0.003)),
    Algorithm("Exponential Decay", "λ=0.01", exponential_decay(λ=0.01)),
    Algorithm("Polynomial Decay", "α=0.5, β=1", polynomial_decay(α=0.5, β=1)),
    Algorithm("Armijo", "α=1, q=0.5, c=1e-4", armijo_rule_gen(α=1, q=0.5, c=1e-4)),
    Algorithm("Wolfe Rule", "α=0.5, c1=1e-4, c2=0.3", wolfe_rule_gen(α=0.5, c1=1e-4, c2=0.3)),
    Algorithm("SciPy Armijo", "!", scipy_armijo),
    Algorithm("SciPy Wolfe", "!", scipy_wolfe),
    Algorithm("Dichotomy", "a=0.0, b=1.0, c=0.5", dichotomy_gen(a=0.0, b=1.0)),
]

def example_table(func: NaryFunc, start: Vector, descent: Descent) -> PrettyTable:
    table = PrettyTable()
    table.field_names = (
        ["Method"]
        + ["Params"]
        + ["x" + str(i + 1) for i in range(len(start))]
        + ["Gradient count"]
        + ["Hessian count"]
        + ["Steps"]
    )
    table.add_rows(
        sorted(
            [algorithm.get_data(func, start, descent) for algorithm in KNOWN],
            key=lambda x: (x[-1], func(x[2:-3])),
        )
    )
    return table

def quadratic(x: float, y: float) -> float:
    return x * x + y * y

def spherical(x: float, y: float) -> float:
    return 100 - np.sqrt(100 - x ** 2 - y ** 2)

def rosenbrock(x: float, y: float) -> float:
    return 0.1 * (1 - x) ** 2 + 0.1 * (y - x ** 2) ** 2

def himmelblau(x: float, y: float) -> float:
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def noise(x: float, y: float, amplitude: float = 0.1) -> float:
    return amplitude * (np.sin(10 * x + 20 * y) + np.cos(15 * x - 10 * y)) / 2

def random_noise(x: float, y: float, amplitude: float = 0.1) -> float:
    return amplitude * np.random.randn()

def noisy_function(
    x: float,
    y: float,
    amplitude: float,
    function: Callable[[float, float], float]
) -> float:
    return function(x, y) + random_noise(x, y, amplitude)

def noisy_wrapper(x: float, y: float) -> float:
    return noisy_function(x, y, amplitude=0.001, function=quadratic)

INTERESTING = [
    [spherical, [-3.0, 2.0], "Quadratic function: 100 - np.sqrt(100 - x^2 - y^2)"],
    [rosenbrock, [0.0, 5.0], "Rosenbrock function: 0.1(1 - x)^2 + 0.1(y - x^2)^2"],
    [himmelblau, [1.0, 1.0], "Himmelblau function: (x^2 + y - 11)^2 + (x + y^2 - 7)^2"],
]

if __name__ == "__main__":
    #descent = gradient_descent
    #func = NaryFunc(quadratic)
    #start = np.array([1.0, 5.0])
    #rule = constant(0.9)
    #print(example_table(func, start, descent))
    #x_min, g_count, h_count, steps, trajectory = descent(func, start, rule)
    #plot_gradient(func, len(start) == 1, len(start) == 2, trajectory, name="Rosenbroke Function")

    #print("Current rule: " + str(rule).split('.')[0].split()[1])
    #print("Optimal x: " + str(x_min))
    #print("Gradient count: " + str(g_count))
    #print("Hessian count: " + str(h_count))
    #print("Steps: " + str(steps))
    # fetch dataset
    df = pd.read_csv("data/spambase.data", header=None)

    # data (as pandas dataframes)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print(df.loc[10])
