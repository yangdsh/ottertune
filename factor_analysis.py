'''
Created on Jul 4, 2016

@author: dvanaken
'''

import os.path
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis

from .cluster import gap_statistic
from .matrix import Matrix
from .preprocessing import Standardize
from .util import stdev_zero
from common.timeutil import stopwatch

FACTOR_CUTOFF = 5


def run_factor_analysis(paths, savedir):
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

    with stopwatch("fit factor analysis model"):
        # Fit the model to calculate the components
        fa = FactorAnalysis()
        fa.fit(matrix.data)
    components = fa.components_[:FACTOR_CUTOFF].T
 
    ks, Wks, Wkbs, sk = gap_statistic(components)
    G = []
    
    Wds = Wkbs - Wks
    for i in range(len(ks) - 1):
        G.append(Wds[i] - (Wds[i+1]-sk[i+1]))
    G = np.array(G)
    print G
    return

    # Run kmeans for different cluster sizes
    cluster_info = {}
    for n_clusters in range(2,21,2):
        cluster_map = {}
        with stopwatch("cluster for k={}".format(n_clusters)):
            kmeans = KMeans(n_clusters)
            kmeans.fit(components)
            cluster_map['cluster_centers'] = kmeans.cluster_centers_
            cluster_map['labels'] = kmeans.labels_
            cluster_map['inertia'] = kmeans.inertia_
        cluster_info[n_clusters] = cluster_map
    
    # Plot the inertia
    plt_clusters = np.empty(len(cluster_info))
    plt_inertia = np.empty(len(cluster_info))
    for i, (n_clusters, entry) in enumerate(sorted(cluster_info.iteritems())):
        plt_clusters[i] = n_clusters
        plt_inertia[i] = entry['inertia']
    
    fig = plt.figure()
    plt.plot(plt_clusters, plt_inertia)
    plt.xlabel("# Clusters")
    plt.ylabel("Inertia")
    plt.title("# Clusters vs. Inertia")
    fig.canvas.set_window_title(os.path.basename(savedir))
    plt.savefig(os.path.join(savedir, "inertia.pdf"), bbox_inches="tight")
    plt.close()
    
    metric_clusters = {}
    for n_clusters, entry in cluster_info.iteritems():
        
        # For each cluster, calculate the distances of each metric from the
        # cluster center. We use the metric closest to the cluster center.
        mclusters = []
        for i in range(n_clusters):
            metric_labels = matrix.columnlabels[entry['labels'] == i]
            component_rows = components[entry['labels'] == i]
            centroid = np.expand_dims(entry['cluster_centers'][i], axis=0)
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
        pickle.dump(metric_clusters, f)
    
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
