'''
Created on Jul 4, 2016

@author: dvanaken
'''

import numpy as np
import os.path
from scipy.spatial.distance import cdist
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from .cluster import KMeans_, KSelection
from .matrix import Matrix
from .preprocessing import get_shuffle_indices
from .util import stdev_zero, stopwatch

OPT_METRICS = ["99th_lat_ms", "throughput_req_per_sec"]

REQUIRED_VARIANCE_EXPLAINED = 90

def run_factor_analysis(paths, savedir, cluster_range, algorithms):
    import gc

    # Load matrices
    assert len(paths) > 0
    matrices = []
    
    with stopwatch("matrix concatenation"):
        for path in paths:
            matrices.append(Matrix.load_matrix(os.path.join(path,
                                                            "y_data_enc.npz")))
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
        print "matrix shape after filter constant: ", matrix.data.shape
        
        # Scale the data
        standardizer = StandardScaler()
        matrix.data = standardizer.fit_transform(matrix.data)
        
        # Shuffle the data rows (experiments x metrics)
        exp_shuffle_indices = get_shuffle_indices(matrix.data.shape[0])
        matrix.data = matrix.data[exp_shuffle_indices]
    
        # Shrink the cluster range if # metrics < max # clusters
        max_clusters = matrix.data.shape[1] + 1
        if max_clusters < cluster_range[1]:
            cluster_range = (cluster_range[0], max_clusters)

    with stopwatch("factor analysis"):
        # Fit the model to calculate the components
        fa = FactorAnalysis()
        fa.fit(matrix.data)
    fa_mask = np.sum(fa.components_ != 0.0, axis=1) > 0.0
    variances = np.sum(np.abs(fa.components_[fa_mask]), axis=1)
    total_variance = np.sum(variances).squeeze()
    print "total variance: {}".format(total_variance)
    var_exp = np.array([np.sum(variances[:i+1]) / total_variance * 100 \
                        for i in range(variances.shape[0])])
    factor_cutoff = np.count_nonzero(var_exp < REQUIRED_VARIANCE_EXPLAINED) + 1
    factor_cutoff = min(factor_cutoff, 10)
    print "factor cutoff: {}".format(factor_cutoff)
    for i,var in enumerate(variances):
        print i, var, np.sum(variances[:i+1]), np.sum(variances[:i+1]) / total_variance

    components = np.transpose(fa.components_[:factor_cutoff]).copy()
    print "components shape: {}".format(components.shape)
    standardizer = StandardScaler()
    components = standardizer.fit_transform(components)
    
    # Shuffle factor analysis matrix rows (metrics x factors)
    metric_shuffle_indices = get_shuffle_indices(components.shape[0])
    components = components[metric_shuffle_indices]
    component_columnlabels = matrix.columnlabels[metric_shuffle_indices].copy()
    
    kmeans = KMeans_(components, cluster_range)
    kmeans.plot_results(savedir, components, component_columnlabels)
    
    # Compute optimal number of clusters K
    for algorithm in algorithms:
        with stopwatch("compute {} (factors={})".format(algorithm,
                                                        factor_cutoff)):
            kselection = KSelection.new(components, cluster_range,
                                        kmeans.cluster_map_, algorithm)
        print "{} optimal # of clusters: {}".format(algorithm,
                                                    kselection.optimal_num_clusters_)
        kselection.plot_results(savedir)
    
    metric_clusters = {}
    featured_metrics = {}
    for n_clusters, (cluster_centers,labels,_) in kmeans.cluster_map_.iteritems():
          
        # For each cluster, calculate the distances of each metric from the
        # cluster center. We use the metric closest to the cluster center.
        mclusters = []
        mfeat_list = []
        for i in range(n_clusters):
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
            assert len(OPT_METRICS) > 0
            label_mask = np.zeros(metric_labels.shape[0])
            for opt_metric in OPT_METRICS:
                label_mask = np.logical_or(label_mask, metric_labels == opt_metric)
            if np.count_nonzero(label_mask) > 0:
                mfeat_list.extend(metric_labels[label_mask].tolist())
            elif len(metric_labels) > 0:
                mfeat_list.append(metric_labels[0])
        metric_clusters[n_clusters] = mclusters
        featured_metrics[n_clusters] = mfeat_list
    
    for n_clusters, mlist in sorted(featured_metrics.iteritems()):
        savepath = os.path.join(savedir, "featured_metrics_{}.txt".format(n_clusters))
        with open(savepath, "w") as f:
            f.write("\n".join(sorted(mlist)))

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
