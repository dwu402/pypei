from scipy.spatial import distance
import numpy as np

def find_paried_distances(x, y, size=None, normalised=False):
    assert len(x) == len(y)
    if size is None:
        size = len(x)//5
    distances = distance.squareform(distance.pdist(np.array(list(zip(x, y)))))
    np.fill_diagonal(distances, np.inf)
    cluster_distances = np.array([np.mean(d[np.argpartition(d, size)[:size]]) for d in distances])
    if normalised:
        cluster_distances /= sum(cluster_distances)
    return cluster_distances
