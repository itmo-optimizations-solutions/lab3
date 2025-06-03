import numpy as np
from loss import Vector, Matrix
from typing import Callable, Tuple

def MSE(features: Matrix,
        targets: Vector,
        weights: Vector,
        batch: Vector
        ) -> float:
    loss = 0
    m = len(batch)
    for i in batch:
        y_predict = weights[0] + features[i] @ weights[1:]
        loss += (targets[i] - y_predict) ** 2
    loss /= m
    return loss

def L2(features: Matrix,
       targets: Vector,
       weights: Vector, 
       batch: Vector,
       lambda_reg=0.01) -> float:
    addition = lambda_reg * np.sum(weights ** 2)
    return MSE(features, targets, weights, batch) + addition

def L1(features: Matrix,
       targets: Vector,
       weights: Vector, 
       batch: Vector,
       lambda_reg=0.01) -> float:
    addition = lambda_reg * np.sum(np.abs(weights))
    return MSE(features, targets, weights, batch) + addition

def Elastic(features: Matrix,
       targets: Vector,
       weights: Vector, 
       batch: Vector,
       lambda_l1=0.01,
       lambda_l2=0.01) -> float:
    l1_term = lambda_l1 * np.sum(np.abs(weights))
    l2_term = lambda_l2 * np.sum(weights ** 2)
    return MSE(features, targets, weights, batch) + l1_term + l2_term
 
