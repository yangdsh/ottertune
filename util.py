'''
Created on Jul 4, 2016

@author: dvanaken
'''

import numpy as np
import os.path


NEARZERO = 1.e-8

def stdev_zero(data, axis=None):
    mstd = np.expand_dims(data.std(axis=axis), axis=axis)
    return (np.abs(mstd) < NEARZERO).squeeze()

def get_featured_metrics(dbms, benchmark=None):
    from globals import Paths
    
    if dbms == "mysql":
        path = "{}_5.6".format(dbms)
    elif dbms == "postgres":
        path = "{}_9.3".format(dbms)
    else:
        raise Exception("Unknown DBMS: {}".format(dbms))
    
    if benchmark is not None:
        path  += "_{}".format(benchmark)
    
    datadir = os.path.join(Paths.DATADIR,
                           "factor_analysis",
                           path)
    with open(os.path.join(datadir, "DetK_optimal_num_clusters.txt"), "r") as f:
        opt_num_clusters = int(f.read().strip())
    with open(os.path.join(datadir, "featured_metrics_{}.txt".format(opt_num_clusters)), "r") as f:
        metrics = np.array(sorted([l.strip() for l in f.readlines()]))
    return metrics

def get_featured_knobs(dbms, benchmark=None):
    from globals import Paths
    
    if dbms == "mysql":
        path = "{}_5.6".format(dbms)
    elif dbms == "postgres":
        path = "{}_9.3".format(dbms)
    else:
        raise Exception("Unknown DBMS: {}".format(dbms))

    if benchmark is not None:
        path  += "_{}".format(benchmark)
    
    datadir = os.path.join(Paths.DATADIR,
                           "lasso",
                           path)
    with open(os.path.join(datadir, "featured_knobs.txt"), "r") as f:
        knobs = np.array([l.strip() for l in f.readlines()])
    return knobs