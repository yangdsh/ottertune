'''
Created on Oct 24, 2017

@author: dva
'''

import contextlib
import datetime
import numpy as np
from numbers import Number

from .matrix import Matrix

NEARZERO = 1.e-8

def stdev_zero(data, axis=None):
    mstd = np.expand_dims(data.std(axis=axis), axis=axis)
    return (np.abs(mstd) < NEARZERO).squeeze()

def get_datetime():
    return datetime.datetime.utcnow()

class TimerStruct():
    
    def __init__(self):
        self.__start_time = 0.0
        self.__stop_time = 0.0
        self.__elapsed = None
    
    @property
    def elapsed_seconds(self):
        if self.__elapsed is None:
            return (get_datetime() - self.__start_time).total_seconds()
        return self.__elapsed.total_seconds()
    
    def start(self):
        self.__start_time = get_datetime()
    
    def stop(self):
        self.__stop_time = get_datetime()
        self.__elapsed = (self.__stop_time - self.__start_time)

@contextlib.contextmanager
def stopwatch(message=None):
    ts = TimerStruct()
    ts.start()
    try:
        yield ts
    finally:
        ts.stop()
        if message is not None:
            print('Total elapsed_seconds time for %s: %.3fs' % (message, ts.elapsed_seconds))

def get_data_base(arr):
    """For a given Numpy array, finds the
    base array that "owns" the actual data."""
    base = arr
    while isinstance(base.base, np.ndarray):
        base = base.base
    return base

def arrays_share_data(x, y):
    return get_data_base(x) is get_data_base(y)

def array_tostring(arr):
    arr_shape = arr.shape
    arr = arr.ravel()
    arr = np.array([str(a) for a in arr])
    return arr.reshape(arr_shape)

def is_numeric_matrix(matrix):
    assert matrix.size > 0
    return isinstance(matrix.ravel()[0], Number)

def is_lexical_matrix(matrix):
    assert matrix.size > 0
    return isinstance(matrix.ravel()[0], str)

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


