"""
Submission file for Hackathon 4.
Group: 2

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
    dictionnaire = {}
    return dictionnaire


def delta_euclidian(a, b, weights=None):
    # Same as `scipy.spatial.distance.euclidean`
    if weights is None:
        weights = np.ones(len(a))
    return sum(weights * (a - b) ** 2)


def delta_mode(a, b, weights=None):
    if weights is None:
        weights = np.ones(len(a))
    return sum(weights * (a != b))


# Your metric
def distance_metric(a, b, **kwargs):
    """
    A pairwise distance between vectors a and b.

    :param a: a vector
    :param b: a vector
    :param kwargs: any keyword arguments you would like to add..
    """
    #########################################################################################################
    # Start : Student version
    #########################################################################################################

    sum_euclidian = 0
    sum_mode = 0

    list_euclidian_a = []
    list_mode_a = []
    list_euclidian_b = []
    list_mode_b = []

    for i in range(len(a)):
        if type(a[i]) == np.float64:
            list_euclidian_a.append(a[i])
            list_euclidian_b.append(b[i])
        else:
            list_mode_a.append(a[i])
            list_mode_b.append(b[i])

    if kwargs.get("weights") is None:
        param = np.ones(len(a))
    else:
        param = kwargs.get("weights")
    if (len(list_mode_a) != 0):
        sum_mode = delta_mode(np.array(list_mode_a), np.array(list_mode_b), weights=param)
    if (len(list_euclidian_a) != 0):
        sum_euclidian = delta_euclidian(np.array(list_euclidian_a), np.array(list_euclidian_b), None)

    return sum_euclidian + sum_mode



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
    #########################################################################################################
    # Start : Student version
    #########################################################################################################

    n_points = len(X)
    dim = len(X[0])

    ## usefull functions
    def nearest_centroid(i, centroids):
        # retourne l' index du centroid le plus proche
        index = 0
        distance_min = metric(X[i], centroids[0])
        for j in range(1, k):
            distance = metric(X[i], centroids[j])
            if (distance < distance_min):
                distance_min = distance
                index = j
        return index

    def clustering_points(centroids):
        # attribue chaque point au centroide le plus proche
        new_cluster_labels = np.zeros(n_points)
        for i in range(n_points):
            new_cluster_labels[i] = nearest_centroid(i, centroids)
        return new_cluster_labels

    def new_centroid(cluster):
        # nouveau centroid= nouveau point avec pour composante la moyenne des composante des points x
        # assignés au cluster
        centroid = [0] * dim
        for i in range(dim):
            if type(X[0][i]) == np.float64:
                s = 0
                for j in cluster:
                    s += X[j][i]
                centroid[i] = s / cluster.size
            else:
                list = [X[m][i] for m in cluster]
                dict = {}
                count = 0
                most_occurent = 0
                for j in list:
                    dict[j] = dict.get(j, 0) + 1
                    if dict[j] >= count:
                        count = dict[j]
                        most_occurent = j
                centroid[i] = most_occurent
        return centroid

    def update_centroids(clusters):
        # mets a jour les k centroids
        centroids = [0] * k
        for i in range(k):
            cluster_i = np.where(clusters == i)[0]
            centroids[i] = new_centroid(cluster_i)
        return centroids

    def check(centroids, old_centroids):
        for i in range(len(centroids)):
            for j in range(dim):
                if centroids[i][j] != old_centroids[i][j]:
                    return True
        return False

    ### initialisation
    index_init_centroids = np.random.choice(np.arange(0, n_points), k, replace=False)
    centroids = [X[i] for i in index_init_centroids]
    old_centroids = centroids
    i = 0
    cluster_labels = np.ones(n_points) * (-1)
    run = True

    ## k-means
    while run and i < n_max:
        cluster_labels = clustering_points(centroids)
        centroids = update_centroids(cluster_labels)

        # run conditions
        run = check(centroids, old_centroids)

        # increment
        old_centroids = centroids
        i += 1

        centroids = np.array(centroids)

    #########################################################################################################
    # End : Student version
    #########################################################################################################

    # Return the results

    return cluster_labels, centroids
