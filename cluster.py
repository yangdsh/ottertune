'''
Created on Jul 4, 2016
 
@author: dva
'''

import os.path
import numpy as np
from sklearn.cluster import KMeans

class KMeansHelper(object):
    
    def __init__(self, X, cluster_range=(1,21)):
        self.cluster_range_ = cluster_range
        self.cluster_map_ = {}
        clusters = range(*self.cluster_range_)
        for k in clusters:
            model = KMeansHelper.fit_model(X, k)
            self.cluster_map_[k] = KMeansHelper.get_model_attributes(model)
        
#         ks, logWks, logWkbs, sk = KMeansHelper.gap_statistic(X,
#                                                              self.cluster_map_,
#                                                              self.cluster_range_)
#         self.ks_      = ks
#         self.logWks_  = logWks
#         self.logWkbs_ = logWkbs
#         self.sk_      = sk
#         self.khats_   = KMeansHelper.compute_khats(self.ks_, self.logWks_,
#                                                    self.logWkbs_, self.sk_)
#         self.opt_cluster_ = ks[:-1][self.khats_ > 0]
#         self.opt_cluster_ = self.opt_cluster_[self.opt_cluster_ > 4]
#         if len(self.opt_cluster_) > 0:
#             self.opt_cluster_ = self.opt_cluster_[0]
#         else:
#             self.opt_cluster_ = -1.0
#         print "optimal cluster: {} (f = {})".format(self.opt_cluster_, X.shape[1])
#         self.distortions_ = np.array([KMeansHelper.compute_distortion(c, X, item[0], item[1]) \
#                                       for c,item in self.cluster_map_.iteritems()])
#         self.weights_ = KMeansHelper.compute_weights(clusters,
#                                                      X.shape[1])
#         self.functs_ = KMeansHelper.compute_eval_function(self.weights_,
#                                                           self.distortions_,
#                                                           clusters)
        self.functs_ = np.empty(len(clusters))
        self.distortions_ = np.empty(len(clusters))
        for i,(k,items) in enumerate(self.cluster_map_.iteritems()):
            self.functs_[i], self.distortions_[i] = KMeansHelper.fK(X, k, items[0], items[1])
        print "optimal cluster: {} (f = {})".format(clusters[np.argmin(self.functs_).squeeze()],
                                                    X.shape[1])
 
    @staticmethod
    def bounding_box(X):
#         ranges = []
#         for i in range(X.shape[1]):
#             xmin, xmax = min(X,key=lambda a:a[i])[i], max(X,key=lambda a:a[i])[i]
#             ranges.append((xmin, xmax))
#         return ranges
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        return mins, maxs
#         xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
#         ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
#         return (xmin,xmax), (ymin,ymax)
  
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
    def gap_statistic(X, cluster_map, cluster_range, B=100):
        #(xmin,xmax), (ymin,ymax) = KMeansHelper.bounding_box(X)
        mins, maxs = KMeansHelper.bounding_box(X)
        #ranges = KMeansHelper.bounding_box(X)
        
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
                Xb = np.empty(X.shape)
                for j in range(X.shape[1]):
                    #Xb[:,j] = np.random.uniform(ranges[j][0], ranges[j][1], size=X.shape[0])
                    Xb[:,j] = np.random.uniform(mins[j], maxs[j], size=X.shape[0])
#                 Xb = []
#                 for _ in range(X.shape[0]):
#                     Xb.append([random.uniform(xmin,xmax),
#                               random.uniform(ymin,ymax)])
#                 Xb = np.array(Xb)
                Xb_model = KMeansHelper.fit_model(Xb, k)
                mu,cluster_labels,_ = KMeansHelper.get_model_attributes(Xb_model)
                logBWkbs[i] = np.log(KMeansHelper.Wk(Xb, mu, cluster_labels))
            logWkbs[indk] = sum(logBWkbs)/B
            sk[indk] = np.sqrt(sum((logBWkbs-logWkbs[indk])**2)/B)
        sk = sk*np.sqrt(1+1/B)
        return (ks, logWks, logWkbs, sk)
    
    def plot(self, savedir):
        clusters = range(*self.cluster_range_)
        inertias = np.empty(len(clusters))
        for i,(_,entry) in enumerate(sorted(self.cluster_map_.iteritems())):
            inertias[i] = entry[2] 
#         KMeansHelper.plot_gaps(self.ks_, self.logWks_,
#                                self.logWkbs_, self.khats_,
#                                inertias, savedir)
        KMeansHelper.plot_functs(clusters,
                                 self.functs_,
                                 inertias,
                                 savedir)

    @staticmethod
    def get_model_attributes(model):
        return model.cluster_centers_, model.labels_, model.inertia_
    
    @staticmethod
    def fit_model(X, K):
        kmeans = KMeans(K, n_init=50)
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
        
    @staticmethod
    def plot_functs(clusters, functs, inertias, savedir):
        import matplotlib.pyplot as plt
        
        # Plot the calculated gap
        plt.figure()
        plt.plot(clusters, functs)
        plt.xlabel("# of clusters K")
        plt.ylabel("F(K)")
        plt.savefig(os.path.join(savedir, "Fks"))
        plt.close()
        
        # Plot the inertia
        plt_clusters = np.empty(len(clusters))
        plt_inertia = np.empty(len(clusters))
        for i, (n_clusters, inertia) in enumerate(zip(clusters,inertias)):
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
    
    @staticmethod
    def compute_distortion(k, X, cluster_centers, cluster_labels):
#         Is = np.empty(k)
#         for i in range(k):
#             Xk = X[cluster_labels == i]
#             #Is[i] = [euclidean(x, cluster_centers[i])**2 for x in Xk]
#             Is[i] = [np.linalg.norm(cluster_centers[i]-x)**2 for x in Xk]
        return sum([np.linalg.norm(cluster_centers[i]-x)**2 \
                    for i in range(k) \
                    for x in X[cluster_labels == i]])
        #return np.sum(Is)
    
    @staticmethod
    def compute_weights(cluster_range, n_dimensions):
        assert n_dimensions > 1
        weights = np.empty(len(cluster_range))
        for i,k in enumerate(cluster_range):
            if k == 1:
                weights[i] = 1.0
            elif k == 2:
                weights[i] = 1.0 - 3.0 / 4.0 * n_dimensions
            else:
                assert k > 2
                assert i >= 1
                weights[i] = weights[i-1] + (1 - weights[i-1]) / 6.0
            #assert weights[i] >= 0.0 and weights[i] <= 1.0
        return weights
    
    @staticmethod
    def compute_eval_function(weights, distortions, cluster_range):
        Fks = np.empty(len(cluster_range))
        for i,k in enumerate(cluster_range):
            if k == 1:
                Fks[i] = 1.0
            else:
                assert i >= 1
                assert k > 1
                if distortions[i-1] != 0.0:
                    Fks[i] = distortions[i] / weights[i]*distortions[i-1]
                else:
                    Fks[i] = 1.0
        return Fks
    
    @staticmethod
    def fK(X, thisk, cluster_centers, cluster_labels, Skm1=0):
        Nd = len(X[0])
        a = lambda k, Nd: 1 - 3/(4*Nd) if k == 2 else a(k-1, Nd) + (1-a(k-1, Nd))/6
#         self.find_centers(thisk, method='++')
#         mu, clusters = self.mu, self.clusters
        Sk = sum([np.linalg.norm(cluster_centers[i]-c)**2 \
                 for i in range(thisk) for c in X[cluster_labels == i]])
        if thisk == 1:
            fs = 1
        elif Skm1 == 0:
            fs = 1
        else:
            fs = Sk/(a(thisk,Nd)*Skm1)
        return fs, Sk      
        
                
                
        
        
