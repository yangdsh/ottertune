'''
Created on Jul 11, 2016

@author: dvanaken
'''
import os.path
import gc
import glob
#import matlab.engine
import numpy as np
import operator
from scipy.spatial.distance import euclidean

from analysis.gaussian_process import predict
from analysis.matrix import Matrix
from analysis.preprocessing import Bin
from analysis.util import get_unique_matrix
from experiment import ExpContext, TunerContext, matlab_engine
from globals import Paths
from common.timeutil import stopwatch

class WorkloadMapper(object):
    
    def __init__(self):
        exp = ExpContext()
        
        
        dbms = exp.dbms.name
        cluster = exp.server.instance_type
        workload_dirs = glob.glob(os.path.join(Paths.DATADIR,
                                               "analysis*{}*{}*".format(dbms,
                                                                        cluster)))
        workload_dirs = [w for w in workload_dirs if "session" not in w]
        assert len(workload_dirs) > 0
        self.workload_dirs_ = workload_dirs
        self.prep_matrices()
        
#         # Filter all matrices by featured metrics. Populate counter that keeps
#         # track of 'high density' experiments (common experiments executed by
#         # different workloads)
#         self.matrices_ = {}
#         for wd in workload_dirs:
#             Xpath = os.path.join(wd, "X_data_unique_{}.npz".format(tuner.num_knobs))
#             ypath = os.path.join(wd, "y_data_unique_{}.npz".format(tuner.num_knobs))
#             X = Matrix.load_matrix(Xpath)
#             y = Matrix.load_matrix(ypath)
#             assert np.array_equal(X.columnlabels, tuner.featured_knobs)
#             y = y.filter(tuner.featured_metrics, "columns")
#             self.matrices_[wd] = (y, X.data)
#         
#         ys = Matrix.vstack([v[0] for v in self.matrices_.values()],
#                            require_equal_columnlabels=True)
#         
#         # Determine deciles for the combined matrix data
#         self.binner_ = Bin(0, axis=0)
#         self.binner_.fit(ys.data)
# 
#         # Bin the metrics using the pre-calculated deciles
#         for wkld in self.matrices_.keys():
#             binned_mtx = self.binner_.transform(self.matrices_[wkld][0].data)
#             assert np.all(binned_mtx >= 0) and np.all(binned_mtx < 10)
#             self.matrices_[wkld][0].data = binned_mtx
#         gc.collect()
    
    
    def prep_matrices(self):
        tuner = TunerContext()
        
        # Filter all matrices by featured metrics. Populate counter that keeps
        # track of 'high density' experiments (common experiments executed by
        # different workloads)
        self.matrices_ = {}
        self.tuned_for_ = tuner.num_knobs
        for wd in self.workload_dirs_:
            Xpath = os.path.join(wd, "X_data_unique_{}.npz".format(tuner.num_knobs))
            ypath = os.path.join(wd, "y_data_unique_{}.npz".format(tuner.num_knobs))
            X = Matrix.load_matrix(Xpath)
            y = Matrix.load_matrix(ypath)
            assert np.array_equal(X.columnlabels, tuner.featured_knobs)
            y = y.filter(tuner.featured_metrics, "columns")
            self.matrices_[wd] = (y, X.data)
        
        ys = Matrix.vstack([v[0] for v in self.matrices_.values()],
                           require_equal_columnlabels=True)
        
        # Determine deciles for the combined matrix data
        self.binner_ = Bin(0, axis=0)
        self.binner_.fit(ys.data)

        # Bin the metrics using the pre-calculated deciles
        for wkld in self.matrices_.keys():
            binned_mtx = self.binner_.transform(self.matrices_[wkld][0].data)
            assert np.all(binned_mtx >= 0) and np.all(binned_mtx < 10)
            self.matrices_[wkld][0].data = binned_mtx
        gc.collect()
            

    def map_workload(self, X_client, y_client):
        tuner = TunerContext()

        # Filter be featured knobs & metrics
        X_client = X_client.filter(tuner.featured_knobs, "columns")
        y_client = y_client.filter(tuner.featured_metrics, "columns")
        
        # Generate unique X,y matrices
        X_client, y_client = get_unique_matrix(X_client, y_client)
        
        # Bin the metrics using the precomputed deciles
        binned_y = self.binner_.transform(y_client.data)

        wkld_scores = {}
        if tuner.incremental_knob_selection and tuner.num_knobs != self.tuned_for_:
            self.prep_matrices()
        with matlab_engine() as engine:
            for wkld,(y_mtx, X_data) in self.matrices_.iteritems():
                wkld_score = 0.0
                row_indices = [(ci,i) for i,l in enumerate(y_mtx.rowlabels) \
                               for ci,cl in enumerate(y_client.rowlabels) \
                               if l == cl]
                missing_indices = [i for i in range(len(y_client.rowlabels)) \
                                   if i not in row_indices]
                if len(row_indices) > 0:
                    wkld_score += np.sum([euclidean(binned_y[ci], y_mtx.data[i]) \
                                          for ci,i in row_indices])
                
                if len(missing_indices) > 0:
                    # Predict any missing indices
                    X_test = X_client.data[missing_indices]
                    ridge = 0.000001 * np.ones(X_data.shape[0])
                    predictions = []
                    for ycol in y_mtx.data.T:
                        # Make predictions
                        ypreds, _, _ = predict(X_data, ycol, X_test, ridge, engine)
                        ypreds = np.array(np.array(ypreds, dtype=int), dtype=float)
                        ypreds[ypreds > 9] = 9
                        ypreds[ypreds < 0] = 0
                        predictions.append(ypreds.ravel())
                    
                    # Create new matrix out of predictions
                    ypreds = np.vstack(predictions).T
                    ypreds = Matrix(ypreds, y_client.rowlabels[missing_indices],
                                    y_client.columnlabels)
    
                    # Update y matrix and X data with new predictions
                    new_y_mtx = Matrix.vstack([y_mtx, ypreds], require_equal_columnlabels=True)
                    new_X_data = np.vstack([X_data, X_test])
                    self.matrices_[wkld] = (new_y_mtx, new_X_data)
                    
                    # Update score using new predictions
                    wkld_score += np.sum([euclidean(u, v) for u,v in \
                                          zip(binned_y[missing_indices],
                                              ypreds.data)])
                wkld_scores[wkld] = wkld_score / X_client.data.shape[0]

        winners = sorted(wkld_scores.items(), key=operator.itemgetter(1))
        print "\nWINNERs:\n{}\n".format(winners)
        
        if tuner.map_to == "best":
            if ExpContext().benchmark.bench_name == "tpch":
                winner = [w for w in self.matrices_.keys() if "tpch" in w]
                assert len(winner) == 1
                winner = winner[0]
            else:
                winner = winners[0][0]
        else:
            winner = winners[-1][0]
        
        print "Mapping to {}\n\n".format(winner)
        return winner
