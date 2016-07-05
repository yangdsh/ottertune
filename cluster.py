'''
Created on Jul 4, 2016
 
@author: dva
'''

import numpy as np
import random
from sklearn.cluster import KMeans
 
def bounding_box(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)
  
def Wk(X, mu, cluster_labels):
    K = len(mu)
    norms = []
    for i in range(K):
        #for j in cluster_labels[i]:
        c = X[cluster_labels == i]
        norms.append(np.linalg.norm(mu[i]-c)**2/(2*len(c)))
    return sum(norms)
   
def gap_statistic(X):
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    # Dispersion for real distribution
    ks = range(1,21)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    for indk, k in enumerate(ks):
        mu, cluster_labels = find_centers(X,k)
        Wks[indk] = np.log(Wk(X, mu, cluster_labels))
        # Create B reference datasets
        B = 10
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = []
            for _ in range(len(X)):
                Xb.append([random.uniform(xmin,xmax),
                          random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            mu, cluster_labels = find_centers(Xb,k)
            BWkbs[i] = np.log(Wk(Xb, mu, cluster_labels))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk)
   
def find_centers(X, K):
    kmeans = KMeans(K)
    kmeans.fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
