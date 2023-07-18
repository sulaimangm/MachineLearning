from typing import List
import numpy as np

seed: int = int(42)

# Function to get covariance matrix


def applyseed(func, seed: int):
    def inner(*args, **kwargs):
        np.random.seed(seed)
        return func(*args, **kwargs)
    return inner


def setseed(s: int):
    global seed
    seed = s


def getCovarianceMatrix(covpre):
    cov = (np.dot(covpre, covpre.transpose())) 
    return cov


@applyseed(seed)
def generateCovMatrix(mean: float, sd: float, size: int):
    covpre = mean + sd * np.random.randn(size, size)
    return getCovarianceMatrix(covpre)


@applyseed(seed)
def generateRandomData(mean: List[int], cov, size: int):
    return np.random.multivariate_normal(mean, cov, size)


def combiningDataFor2Classes(size: int, x1, x2):
    X = np.concatenate((x1, x2))
    Xc = np.ones(size)
    Xc = np.concatenate((Xc, np.zeros(size)))
    return (X, Xc)
