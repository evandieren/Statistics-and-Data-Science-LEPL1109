"""
Submission file for Hackathon 4.
Group: <group_number>

Please, make sure this file passes tests from `validate.py` before submitting!

Do not:
- change the signature and names of functions
- import packages that where not installed during the Install session

You can:
- create any number of functions, methods, and use them inside the functions that will be assessed

Warning: plagiarism is not tolerated and we can detect it
"""

from typing import Any, Callable, Dict, Union, Tuple
import numpy as np
import pandas as pd


def generate_metric_input_vector() -> np.ndarray:
    """
    Generate a random input that can be passed to your distance metric.
    Usually, this will be a vector of random floats.

    However, as your metric can accept vectors of different types,
    you can change this function to generate an appropriate vector for your metric.

    Notes: we will use this function for testing your implementation.

    :return: a vector
    """
    return np.random.rand(10)


def generate_metric_input_kwargs() -> Dict[str, Any]:
    """
    Generate kwargs (keyword arguments) for the distance metric.

    Change this function only if needed.

    Notes: we will use this function for testing your implementation.

    :return: a dictionary of kwargs for the metric
    """
    return dict()


def distance_metric(a: np.ndarray, b: np.ndarray, **kwargs: Any) -> float:
    """
    A pairwise distance between vectors a and b.

    :param a: a vector
    :param b: a vector
    :param kwargs: any keyword arguments you would like to add..
    :return: a scalar
    """
    raise NotImplementedError


def generate_KPrototypes_input_matrix() -> Union[np.ndarray, pd.DataFrame]:
    """
    Generate a random input that can be passed to your K-Prototypes.
    Usually, this will be a matrix of random floats.

    However, as your K-Prototypes can accept matrices with arbitrary types,
    you can change this function to generate an appropriate matrix for your function.

    Notes: we will use this function for testing your implementation.

    :return: a matrix or DataFrame
    """
    return np.random.rand(100, 10)


def KPrototypes(
    X: Union[np.ndarray, pd.DataFrame],
    k: int = 5,
    n_max: int = 100,
    metric: Callable = distance_metric,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a K-Prototypes algorithm on input data.

    :param X: m-by-n observation matrix or DataFrame
    :param k: number of clusters
    :param n_max: maximum number of iterations
    :param metric: the distance metric to use
    :param kwargs: optional arguments to be passed to `metric`
    :return: the cluster labels and the centroids
    """
    raise NotImplementedError
