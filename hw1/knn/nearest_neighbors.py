from typing import Tuple
import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    raise NotImplementedError()


class NearestNeighborsFinder:
    def __init__(self, n_neighbors: int, metric: str = "euclidean") -> None:
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
            self.less_is_better = True
        elif metric == "cosine":
            self._metric_func = cosine_distance
            self.less_is_better = False
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False) -> Tuple[np.array]:
        distances = self._metric_func(X, self._X)
        sorted_indices = np.argsort(distances)
        if self.less_is_better:
            neighbours_indices = sorted_indices[:, :self.n_neighbors] # smaller distance -> first n indices
        else:
            neighbours_indices = sorted_indices[:, -self.n_neighbors:] # higher similarity -> last n indices

        if return_distance:
            return distances, neighbours_indices
        else:
            return neighbours_indices
