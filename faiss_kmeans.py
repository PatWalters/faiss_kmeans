#!/usr/bin/env python

import sys
import faiss
import h5py
import csv
from functools import wraps
from time import time


def timing(f):
    """
    Decorator to measure execution time, adapted from
    # https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
    # https://codereview.stackexchange.com/questions/169870/decorator-to-measure-execution-time-of-a-function
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f.__name__, f"Elapsed time: {end - start:.2f} sec")
        return result

    return wrapper


@timing
def get_centers(num_centroids, dist, cluster_id):
    """
    Find the most central molecule in each cluster
    :param num_centroids: number of cluster centers
    :param dist: list of distances from the cluster center
    :param cluster_id: list of cluster identifiers
    :return:
    """
    nearest = [[sys.float_info.max, -1]] * num_centroids
    for i, (d, c) in enumerate(zip(dist, cluster_id)):
        if d < nearest[c][0]:
            nearest[c] = [d, i]
    return nearest


@timing
def faiss_kmeans(infile, ncentroids, niter=20):
    """
    K-Means clustering with FAISS
    :param infile: Input file name
    :param ncentroids: desired number of clusters
    :param niter: maximum number of iterations
    :return: None
    """
    h5f = h5py.File(infile, 'r')
    x = h5f['fp_list'][:]
    smiles_list = h5f['smiles_list'][:]
    name_list = h5f['name_list'][:]
    h5f.close()

    verbose = True
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(x)
    D, I = kmeans.index.search(x, 1)
    writer = csv.writer(open("detail.csv", "w"))
    writer.writerow(["SMILES", "NAME", "DIST", "CLUSTER"])
    for smiles, name, d, i in zip(smiles_list, name_list, [x[0] for x in D], [x[0] for x in I]):
        writer.writerow([smiles[0].decode('utf-8'), name[0].decode('utf-8'), d, i])
    dist_list = get_centers(ncentroids, [x[0] for x in D], [x[0] for x in I])
    ofs = open("centers.smi", "w")
    for cluster_idx, (_, c) in enumerate(dist_list, 1):
        print(smiles_list[c][0].decode('utf-8'), name_list[c][0].decode('utf-8'), cluster_idx, file=ofs)
    ofs.close()


@timing
def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} infile num_clusters")
        sys.exit(1)
    faiss_kmeans(sys.argv[1], int(sys.argv[2]))


if __name__ == "__main__":
    main()
