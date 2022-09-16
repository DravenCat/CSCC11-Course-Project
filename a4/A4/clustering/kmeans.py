"""
CSCC11 - Introduction to Machine Learning, Winter 2022, Assignment 4
B. Chan, S. Wei, D. Fleet
"""
import time

import numpy as np

class KMeans:
    def __init__(self, init_centers):
        """ This class represents the K-means model.

        TODO: You will need to implement the methods of this class:
        - train: ndarray, int -> ndarray

        Implementation description will be provided under each method.

        For the following:
        - N: Number of samples.
        - D: Dimension of input features.
        - K: Number of centers.
             NOTE: K > 1

        Args:
        - init_centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional centers.
        """

        assert len(init_centers.shape) == 2, f"init_centers should be a KxD matrix. Got: {init_centers.shape}"
        (self.K, self.D) = init_centers.shape
        assert self.K > 1, f"There must be at least 2 clusters. Got: {self.K}"

        # Shape: K x D
        self.centers = np.copy(init_centers)

    def train(self, train_X, max_iterations=1000):
        """ This method trains the K-means model.

        NOTE: This method updates self.centers

        The algorithm is the following:
        - Assigns data points to the closest cluster center.
        - Re-computes cluster centers based on the data points assigned to them.
        - Update the labels array to contain the index of the cluster center each point is assigned to.
        - Loop ends when the labels do not change from one iteration to the next. 

        Args:
        - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.
        - max_iterations (int): Maximum number of iterations.

        Output:
        - labels (ndarray (shape: (N, 1))): A N-column vector consisting N labels of input data.
        """
        assert len(train_X.shape) == 2 and train_X.shape[1] == self.D, f"train_X should be a NxD matrix. Got: {train_X.shape}"
        assert max_iterations > 0, f"max_iterations must be positive. Got: {max_iterations}"
        N = train_X.shape[0]

        labels = np.empty(shape=(N, 1), dtype=np.long)
        distances = np.empty(shape=(N, self.K))
        for _ in range(max_iterations):
            old_labels = np.copy(labels)

            # ====================================================
            # TODO: Implement your solution within the box
            # labeling update
            for i in range(N):
                for k in range(self.K):
                    distances[i][k] = np.linalg.norm(train_X[i] - self.centers[k])
            for i in range(N):
                labels[i] = np.argmin(distances[i])

            # update centers
            for j in range(self.K):
                s = np.zeros(shape=(train_X.shape[1],), dtype=np.long)
                for i in range(N):
                    if labels[i] == j:
                        s = s + train_X[i]
                self.centers[j] = s / np.sum(labels == j)
            # ====================================================
            # Check convergence
            if np.allclose(old_labels, labels):
                break

        return labels
