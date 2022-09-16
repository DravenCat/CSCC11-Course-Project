"""
CSCC11 - Introduction to Machine Learning, Winter 2022, Assignment 4
B. Chan, S. Wei, D. Fleet
"""

import numpy as np

def get_closest_distance(point, centers, K):
    min_distance = np.inf
    for i in range(K):
        distance = np.linalg.norm(centers[i] - point)
        if distance < min_distance:
            min_distance = distance
    return min_distance

def kmeans_pp(K, train_X):
    """ This function runs K-means++ algorithm to choose the centers.

    Args:
    - K (int): Number of centers.
    - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.

    Output:
    - centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional centers.
    """
    centers = np.empty(shape=(K, train_X.shape[1]))

    # ====================================================
    # TODO: Implement your solution within the box
    # choose first center
    modified_X = np.copy(train_X)
    chosen = np.random.randint(modified_X.shape[0], size=1)
    centers[0] = modified_X[chosen]
    modified_X = np.delete(modified_X, chosen, axis=0)

    # choose remaining centers
    d = np.zeros(shape=(modified_X.shape[0],), dtype=np.long)
    for i in range(1, K):
        total = 0.0
        for j in range(modified_X.shape[0]):
            d[j] = get_closest_distance(modified_X[j], centers, i)
            total += (d[j] * d[j])

        total *= np.random.random()
        x = -1
        while total > 0:
            x += 1
            total -= (d[x] * d[x])
        centers[i] = modified_X[x]
        modified_X = np.delete(modified_X, x, axis=0)
    # ====================================================
    return centers

def random_init(K, train_X):
    """ This function randomly chooses K data points as centers.

    Args:
    - K (int): Number of centers.
    - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.

    Output:
    - centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional centers.
    """
    centers = train_X[np.random.randint(train_X.shape[0], size=K)]
    return centers
