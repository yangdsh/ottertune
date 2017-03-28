'''
Created on Jul 4, 2016

@author: dvanaken
'''

import numpy as np
import os.path
from analysis.matrix import Matrix


NEARZERO = 1.e-8

def stdev_zero(data, axis=None):
    mstd = np.expand_dims(data.std(axis=axis), axis=axis)
    return (np.abs(mstd) < NEARZERO).squeeze()

def get_featured_metrics(dbms, cluster, benchmark=None, num_clusters=None):
    from globals import Paths

    path = "{}_{}".format(dbms, cluster)
    if benchmark is not None:
        path  += "_{}".format(benchmark)

    datadir = os.path.join(Paths.DATADIR,
                           "factor_analysis",
                           path)
    if not os.path.exists(datadir):
        return None
    if num_clusters is None:
        with open(os.path.join(datadir, "DetK_optimal_num_clusters.txt"), "r") as f:
            num_clusters = int(f.read().strip())
    with open(os.path.join(datadir, "featured_metrics_{}.txt".format(num_clusters)), "r") as f:
        metrics = np.array(sorted([l.strip() for l in f.readlines()]))
    return metrics

def get_featured_knobs(dbms, cluster, benchmark=None):
    from globals import Paths

    path = "{}_{}".format(dbms, cluster)
    if benchmark is not None:
        path  += "_{}".format(benchmark)

    datadir = os.path.join(Paths.DATADIR,
                           "lasso",
                           path)
    if not os.path.exists(datadir):
        return None
    with open(os.path.join(datadir, "featured_knobs.txt"), "r") as f:
        knobs = np.array([l.strip() for l in f.readlines()])
    return knobs

def get_unique_matrix(X, y):
    X_unique, unique_indexes = X.unique_rows(return_index=True)
    assert np.array_equal(X_unique.columnlabels, X.columnlabels)
    y_unique = Matrix(y.data[unique_indexes],
                      y.rowlabels[unique_indexes],
                      y.columnlabels)

    rowlabels = np.empty_like(X_unique.rowlabels, dtype=object)
    exp_set = set()
    for i,row in enumerate(X_unique.data):
        exp_label = tuple((l,r) for l,r in zip(X_unique.columnlabels, row))
        assert exp_label not in exp_set
        rowlabels[i] = exp_label
        exp_set.add(exp_label)
    y_unique.rowlabels = rowlabels
    X_unique.rowlabels = rowlabels
    if X_unique.data.shape != X.data.shape:
        print "\n\nDIFF(num_knobs={}): X_unique: {}, X: {}\n\n".format(X_unique.columnlabels.shape[0], X_unique.data.shape, X.data.shape)
        dup_map = {}
        dup_indexes = np.array([d for d in range(X.data.shape[0]) \
                                if d not in unique_indexes])
        for dup_idx in dup_indexes:
            dup_label = tuple((u''+l,r) for l,r in \
                              zip(X_unique.columnlabels,
                                  X.data[dup_idx]))
            primary_idx = [idx for idx,rl in enumerate(rowlabels) \
                           if rl == dup_label]
            assert len(primary_idx) == 1
            primary_idx = primary_idx[0]
            if primary_idx not in dup_map:
                dup_map[primary_idx] = [y_unique.data[primary_idx]]
            dup_map[primary_idx].append(y.data[dup_idx])
        for idx, yvals in dup_map.iteritems():
            y_unique.data[idx] = np.median(np.vstack(yvals), axis=0)
    return X_unique, y_unique

def get_exp_labels(X, columnlabels):
    rowlabels = np.empty(X.shape[0], dtype=object)
    for i,row in enumerate(X):
        exp_label = tuple((l,r) for l,r in zip(columnlabels, row))
        rowlabels[i] = exp_label
    return rowlabels

def get_knob_range(dbms, cluster):
    num_featured_knobs = len(get_featured_knobs(dbms, cluster))
    return np.arange(2,num_featured_knobs + 2,2)

def combine_workloads(paths, savedirs):
    from collections import Counter

    exp_counter = Counter()
    for path in paths:
        y = Matrix.load_matrix(os.path.join(path, "y_data_wkld.npz"))
        exp_counter.update(y.rowlabels)
    print [c for _,c in exp_counter.most_common(10)]

def array_tostring(arr):
    arr_shape = arr.shape
    arr = arr.ravel()
    arr = np.array([str(a) for a in arr])
    return arr.reshape(arr_shape)


