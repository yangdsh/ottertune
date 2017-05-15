'''
Created on Jul 4, 2016
 
@author: dva
'''

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os.path

from abc import ABCMeta, abstractmethod, abstractproperty
from sklearn.cluster import KMeans

class KMeans_(object):
    
    def __init__(self, X, cluster_range, n_init=50):
        self.clusters_ = np.arange(*cluster_range)
        self.cluster_map_ = {}
        for k in self.clusters_:
            model = KMeans_.fit_model(X, k, n_init)
            self.cluster_map_[k] = KMeans_.get_model_attributes(model)

    @staticmethod
    def fit_model(X, K, n_init=50):
        kmeans = KMeans(K, n_init=n_init)
        kmeans.fit(X)
        return kmeans

    @staticmethod
    def get_model_attributes(model):
        return model.cluster_centers_, model.labels_, model.inertia_

    def plot_results(self, savedir, X, X_labels):
        # Plot the inertia
        inertias = np.array([inrt for _,(_,_,inrt) in \
                             sorted(self.cluster_map_.iteritems())])
        fig = plt.figure()
        plt.plot(self.clusters_, inertias)
        plt.xlabel("number of clusters k")
        plt.ylabel("within sum of squares W_k")
        plt.title("within sum of squares vs. number of clusters")
        fig.canvas.set_window_title(os.path.basename(savedir))
        plt.savefig(os.path.join(savedir, "sum_of_squares.pdf"), bbox_inches="tight")
        plt.close()
        
        # Save the cluster map
        with open(os.path.join(savedir, "clusters_map.p"), "w") as f:
            pickle.dump(self.cluster_map_, f)


class KSelection(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, X, cluster_range, cluster_map):
        self.clusters_ = np.arange(*cluster_range)
        self.cluster_map_ = cluster_map
        self.optimal_num_clusters_ = None
    
    @abstractproperty
    def name(self):
        pass
    
    @abstractmethod
    def compute(self):
        pass
    
    @abstractmethod
    def plot_results(self, savedir):
        assert self.optimal_num_clusters_ is not None
        filepath = os.path.join(savedir,
                                "{}_optimal_num_clusters.txt".format(self.name))
        with open(filepath, "w") as f:
            f.write(str(self.optimal_num_clusters_))
    
    @staticmethod
    def new(X, cluster_range, cluster_map, algorithm):
        if algorithm == "DetK":
            return DetK(X, cluster_range, cluster_map)
        elif algorithm == "GapStatistic":
            return GapStatistic(X, cluster_range, cluster_map)
        else:
            raise Exception("Unknown cluster algorithm: {}".format(algorithm))

class GapStatistic(KSelection):
    
    def __init__(self, X, cluster_range, cluster_map, n_B=50):
        super(GapStatistic, self).__init__(X, cluster_range, cluster_map)
        self.n_B_ = n_B
        self.logWks_, self.logWkbs_, self.sk_ = self.compute(X)
        
        len_ks = self.clusters_.shape[0]
        self.khats_ = np.zeros(len_ks)
        gaps = self.logWkbs_ - self.logWks_
        gsks = gaps - self.sk_
        for i in range(1, len_ks):
            self.khats_[i] = gaps[i-1] - gsks[i]
            if self.optimal_num_clusters_ is None and gaps[i-1] >= gsks[i]:
                self.optimal_num_clusters_ = self.clusters_[i-1]
        if self.optimal_num_clusters_ is None:
            self.optimal_num_clusters_ = 0

    @property
    def name(self):
        return "GapStatistic"
 
    @staticmethod
    def bounding_box(X):
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        return mins, maxs
#         xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
#         ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
#         return (xmin,xmax), (ymin,ymax)
  
    @staticmethod
    def Wk(X, mu, cluster_labels):
        K = len(mu)
        return sum([np.linalg.norm(mu[i]-x)**2 \
                    for i in range(K) \
                    for x in X[cluster_labels == i]])
    
    def compute(self, X):
        mins, maxs = GapStatistic.bounding_box(X)
        
        # Dispersion for real distribution
        len_ks = self.clusters_.shape[0]
        logWks = np.zeros(len_ks)
        logWkbs = np.zeros(len_ks)
        sk = np.zeros(len_ks)
        for indk, k in enumerate(self.clusters_):
            mu,cluster_labels,_ = self.cluster_map_[k]
            logWks[indk] = np.log(GapStatistic.Wk(X, mu, cluster_labels))

            # Create B reference datasets
            logBWkbs = np.zeros(self.n_B_)
            for i in range(self.n_B_):
                Xb = np.empty(X.shape)
                for j in range(X.shape[1]):
                    Xb[:,j] = np.random.uniform(mins[j], maxs[j], size=X.shape[0])
                Xb_model = KMeans_.fit_model(Xb, k)
                mu,cluster_labels,_ = KMeans_.get_model_attributes(Xb_model)
                logBWkbs[i] = np.log(GapStatistic.Wk(Xb, mu, cluster_labels))
            logWkbs[indk] = sum(logBWkbs) / self.n_B_
            sk[indk] = np.sqrt(sum((logBWkbs-logWkbs[indk])**2) / self.n_B_)
        sk = sk * np.sqrt(1 + 1 / self.n_B_)
        return logWks, logWkbs, sk

    def plot_results(self, savedir):
        super(GapStatistic, self).plot_results(savedir)
        
        # Plot the calculated gap
        gaps = self.logWkbs_ - self.logWks_
        fig = plt.figure()
        plt.plot(self.clusters_, gaps)
        plt.title("gap vs. number of clusters")
        plt.xlabel("number of clusters k")
        plt.ylabel("gap_k")
        fig.canvas.set_window_title(os.path.basename(savedir))
        plt.savefig(os.path.join(savedir, "gap"),
                    bbox_inches="tight")
        plt.close()
        
        # Plot the sum of squares
        fig = plt.figure()
        plt.plot(self.clusters_, np.exp(self.logWks_))
        plt.xlabel("number of clusters k")
        plt.ylabel("within sum of squares W_k")
        fig.canvas.set_window_title(os.path.basename(savedir))
        plt.savefig(os.path.join(savedir, "gap-sum_of_squares"),
                    bbox_inches="tight")
        plt.close()
        
        # Plot the gap statistic
        fig = plt.figure()
        plt.bar(self.clusters_, self.khats_)
        plt.title("gap statistic vs. number of clusters")
        plt.xlabel("number of clusters K")
        plt.ylabel("gap(k)-(gap(k+1)-s(k+1))")
        fig.canvas.set_window_title(os.path.basename(savedir))
        plt.savefig(os.path.join(savedir, "gap_final"),
                    bbox_inches="tight")
        plt.close()
    

class DetK(KSelection):
    
    def __init__(self, X, cluster_range, cluster_map):
        super(DetK, self).__init__(X, cluster_range, cluster_map)
        self.Fs_, self.Sks_ = self.compute(X)
        self.optimal_num_clusters_ = self.clusters_[np.argmin(self.Fs_)]

    @property
    def name(self):
        return "DetK" 
    
    def compute(self, X):
        Nd = X.shape[1]
        Fs = np.empty(len(self.clusters_))
        Sks = np.empty(len(self.clusters_))
        for i, (n_clusters, (cluster_centers, cluster_labels, _)) \
                in enumerate(sorted(self.cluster_map_.iteritems())):
            a = lambda k, Nd: 1 - 3/(4*Nd) if k == 2 \
                    else a(k-1, Nd) + (1-a(k-1, Nd))/6
            Sks[i] = sum([np.linalg.norm(cluster_centers[j]-c)**2 \
                          for j in range(n_clusters) \
                          for c in X[cluster_labels == j]])
            if n_clusters == 1:
                Fs[i] = 1
            elif Sks[i-1] == 0:
                Fs[i] = 1
            else:
                Fs[i] = Sks[i]/(a(n_clusters, Nd) * Sks[i-1])
        return Fs, Sks

        
    def plot_results(self, savedir):
        super(DetK, self).plot_results(savedir)
        
        # Plot the evaluation function
        fig = plt.figure()
        plt.plot(self.clusters_, self.Fs_)
        plt.xlabel("number of clusters k")
        plt.ylabel("evaluation function F_k")
        plt.title("evaluation function vs. number of clusters")
        fig.canvas.set_window_title(os.path.basename(savedir))
        plt.savefig(os.path.join(savedir, "eval_function"),
                    bbox_inches="tight")
        plt.close()   
        
                
                
        
        
