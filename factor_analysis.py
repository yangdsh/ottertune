'''
Created on Jul 4, 2016

@author: dvanaken
'''

import os.path
import cPickle as pickle
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import FactorAnalysis

from .cluster import KMeansHelper
from .matrix import Matrix
from .preprocessing import Standardize
from .util import stdev_zero
from common.timeutil import stopwatch


def run_factor_analysis(paths, savedir, factor_cutoff=5):
    import gc

    # Load matrices
    assert len(paths) > 0
    matrices = []
    
    with stopwatch("matrix concatenation"):
        for path in paths:
            matrices.append(Matrix.load_matrix(path))
        
        # Combine matrix data if more than 1 matrix
        if len(matrices) > 1:
            matrix = Matrix.vstack(matrices, require_equal_columnlabels=True)
        else:
            matrix = matrices[0]
        del matrices
        gc.collect()
    
    with stopwatch("preprocessing"):
        # Filter out columns with near zero standard deviation
        # i.e., constant columns
        column_mask = ~stdev_zero(matrix.data, axis=0)
        filtered_columns = matrix.columnlabels[column_mask]
        matrix = matrix.filter(filtered_columns, 'columns')
        
        # Scale the data
        standardizer = Standardize()
        matrix.data = standardizer.fit_transform(matrix.data)
        
        # TODO: shuffle the data
        n_rows = matrix.data.shape[0]
        exp_shuffle_indices = np.random.choice(n_rows, n_rows, replace=False)
        matrix.data = matrix.data[exp_shuffle_indices]

    with stopwatch("fit factor analysis model"):
        # Fit the model to calculate the components
        fa = FactorAnalysis()
        fa.fit(matrix.data)
    components = fa.components_[:factor_cutoff].T
    standardizer = Standardize()
    components = standardizer.fit_transform(components)
    
    # Shuffle metrics
    n_metrics = components.shape[0]
    metric_shuffle_indices = np.random.choice(n_metrics, n_metrics, replace=False)
    components = components[metric_shuffle_indices]
    component_columnlabels = matrix.columnlabels[metric_shuffle_indices]
 
    clusterer = KMeansHelper(components)
    clusterer.plot(savedir)
    
    metric_clusters = {}
    for n_clusters, (cluster_centers,labels,_) in clusterer.cluster_map_.iteritems():
         
        # For each cluster, calculate the distances of each metric from the
        # cluster center. We use the metric closest to the cluster center.
        mclusters = []
        for i in range(n_clusters):
            #metric_labels = matrix.columnlabels[labels == i]
            metric_labels = component_columnlabels[labels == i]
            component_rows = components[labels == i]
            centroid = np.expand_dims(cluster_centers[i], axis=0)
            dists = np.empty(component_rows.shape[0])
            for j,row in enumerate(component_rows):
                row = np.expand_dims(row, axis=0)
                dists[j] = cdist(row, centroid, 'euclidean').squeeze()
            order_by = np.argsort(dists)
            metric_labels = metric_labels[order_by]
            dists = dists[order_by]
            mclusters.append((i,metric_labels, dists))
        metric_clusters[n_clusters] = mclusters
    
    with open(os.path.join(savedir, "metric_clusters.p"), "w") as f:
        pickle.dump(clusterer.cluster_map_, f)
    
    for n_clusters, memberships in sorted(metric_clusters.iteritems()):
        cstr = ""
        for i,(cnum, lab, dist) in enumerate(memberships):
            assert i == cnum
            cstr += "---------------------------------------------\n"
            cstr += "CLUSTERS {}\n".format(i)
            cstr += "---------------------------------------------\n\n"
            
            for l,d in zip(lab,dist):
                cstr += "{}\t({})\n".format(l,d)
            cstr += "\n\n"

        savepath = os.path.join(savedir, "membership_{}.txt".format(n_clusters))
        with open(savepath, "w") as f:
            f.write(cstr)
