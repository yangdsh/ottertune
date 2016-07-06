'''
Created on Jul 4, 2016
 
@author: dva
'''

import os.path
import numpy as np
import random
from sklearn.cluster import KMeans

class KMeansHelper(object):
    
    def __init__(self, X, cluster_range=(1,21)):
        self.cluster_range_ = cluster_range
        self.cluster_map_ = {}
        for k in range(*self.cluster_range_):
            model = KMeansHelper.fit_model(X, k)
            self.cluster_map_[k] = KMeansHelper.get_model_attributes(model)
        
        ks, logWks, logWkbs, sk = KMeansHelper.gap_statistic(X,
                                                             self.cluster_map_,
                                                             self.cluster_range_)
        self.ks_      = ks
        self.logWks_  = logWks
        self.logWkbs_ = logWkbs
        self.sk_      = sk
        self.khats_   = KMeansHelper.compute_khats(self.ks_, self.logWks_,
                                                   self.logWkbs_, self.sk_)
 
    @staticmethod
    def bounding_box(X):
        xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
        ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
        return (xmin,xmax), (ymin,ymax)
  
    @staticmethod
    def Wk(X, mu, cluster_labels):
        K = len(mu)
        return sum([np.linalg.norm(mu[i]-x)**2 for i in range(K) for x in X[cluster_labels == i]])
#         K = len(mu)
#         norms = []
#         for i in range(K):
#             c = X[cluster_labels == i]
#             norms.append(np.linalg.norm(mu[i]-c)**2/(2*len(c)))
#         return sum(norms)
    
    @staticmethod
    def gap_statistic(X, cluster_map, cluster_range, B=10):
        (xmin,xmax), (ymin,ymax) = KMeansHelper.bounding_box(X)
        # Dispersion for real distribution
        ks = np.arange(*cluster_range)
        len_ks = ks.shape[0]
        logWks = np.zeros(len_ks)
        logWkbs = np.zeros(len_ks)
        sk = np.zeros(len_ks)
        for indk, k in enumerate(ks):
            mu,cluster_labels,_ = cluster_map[k]
            logWks[indk] = np.log(KMeansHelper.Wk(X, mu, cluster_labels))
            # Create B reference datasets
            logBWkbs = np.zeros(B)
            for i in range(B):
                Xb = []
                for _ in range(X.shape[0]):
                    Xb.append([random.uniform(xmin,xmax),
                              random.uniform(ymin,ymax)])
                Xb = np.array(Xb)
                Xb_model = KMeansHelper.fit_model(Xb, k)
                mu,cluster_labels,_ = KMeansHelper.get_model_attributes(Xb_model)
                logBWkbs[i] = np.log(KMeansHelper.Wk(Xb, mu, cluster_labels))
            logWkbs[indk] = sum(logBWkbs)/B
            sk[indk] = np.sqrt(sum((logBWkbs-logWkbs[indk])**2)/B)
        sk = sk*np.sqrt(1+1/B)
        return (ks, logWks, logWkbs, sk)
    
    def plot(self, savedir):
        inertias = np.empty(self.ks_.shape[0])
        for i,(_,entry) in enumerate(sorted(self.cluster_map_.iteritems())):
            inertias[i] = entry[2] 
        KMeansHelper.plot_gaps(self.ks_, self.logWks_,
                               self.logWkbs_, self.khats_,
                               inertias, savedir)

    @staticmethod
    def get_model_attributes(model):
        return model.cluster_centers_, model.labels_, model.inertia_
    
    @staticmethod
    def fit_model(X, K):
        kmeans = KMeans(K, n_init=20)
        kmeans.fit(X)
        return kmeans
    
    @staticmethod
    def compute_khats(ks, logWks, logWkbs, sk):
        len_ks = ks.shape[0]
        khats = np.empty(len_ks - 1)
        gs = logWkbs - logWks
        gsks = gs - sk
        for i in range(len_ks - 1):
            khats[i] = gs[i] - gsks[i+1]
        return khats
    
    @staticmethod
    def plot_gaps(ks, logWks, logWkbs, khats, inertias, savedir):
        import matplotlib.pyplot as plt
        
        # Plot the calculated gap
        gaps = logWkbs - logWks
        plt.figure()
        plt.plot(ks, gaps)
        plt.xlabel("# of clusters K")
        plt.ylabel("Gap")
        plt.savefig(os.path.join(savedir, "gap"))
        plt.close()
        
        # Plot the sum of squares
        plt.figure()
        plt.plot(ks, np.exp(logWks))
        plt.xlabel("# of clusters K")
        plt.ylabel("per-cluster sum of squares")
        plt.savefig(os.path.join(savedir, "sos"))
        plt.close()
        
        # Plot the gap statistic
        plt.figure()
        plt.bar(ks[:-1], khats)
        plt.xlabel("# of clusters K")
        plt.ylabel("Gap(k)-(Gap(k+1)-s(k+1))")
        plt.savefig(os.path.join(savedir, "gap_final"))
        plt.close()
        
        # Plot the inertia
        plt_clusters = np.empty(len(ks))
        plt_inertia = np.empty(len(ks))
        for i, (n_clusters, inertia) in enumerate(zip(ks,inertias)):
            plt_clusters[i] = n_clusters
            plt_inertia[i] = inertia
         
        fig = plt.figure()
        plt.plot(plt_clusters, plt_inertia)
        plt.xlabel("# Clusters")
        plt.ylabel("Inertia")
        plt.title("# Clusters vs. Inertia")
        fig.canvas.set_window_title(os.path.basename(savedir))
        plt.savefig(os.path.join(savedir, "inertia.pdf"), bbox_inches="tight")
        plt.close()
        
