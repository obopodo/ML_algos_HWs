import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y).flatten()
        self._classes = np.unique(y)
        return self

    def _predict_precomputed(self, indices, distances):
        N = len(indices)
        neighbours_labels = np.array([self._labels[inds] for inds in indices])
        # neighbours_labels = np.ones((1, self._labels.shape[0])).dot(self._labels)
        if self._weights == 'uniform':
            weights = np.ones(indices.shape)
        else:
            weights = distances
        if self._finder.metric == 'cosine':
            weights = 1 - weights
        weights = 1 / weights

        class_weights = np.zeros((N, len(self._classes)))
        for i, c in enumerate(self._classes):
            mask = (neighbours_labels == c).astype(int)
            class_weights[:, i] = (weights * mask).sum(axis=1)

        best_classes = self._classes[class_weights.argmax(axis=1)]        
        return best_classes

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):
    '''
    ?????? ?????????? ???????? ??????????, ???????????? ?????? ???? ?????????? ?????????????????? ?????????????????? ??????????????
    ?? ?????? ?????????? ?????? ?????????????? ???????????? ?????????????? ???? sklearn
    '''

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform', batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self._batch_size = batch_size

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)

        raise NotImplementedError()
