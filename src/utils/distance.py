import numpy as np


def euclidean(A, B):
    return np.linalg.norm(B - A, axis=1)

# TODO: add more distance


def manhattan(A, B):
    return np.linalg.norm(B - A, ord=1, axis=1)


def chebyshev(A, B):
    return np.linalg.norm(B - A, ord=np.inf, axis=1)
