'''
Created on Jul 11, 2016

@author: dvanaken
'''
import os.path
import gc
import glob, zlib
#import matlab.engine
import dill as pickle
import multiprocessing
import numpy as np
import operator
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from time import sleep

#from analysis.gaussian_process import tf_predict
from analysis.gp_tf import GPR
from analysis.matrix import Matrix
from analysis.preprocessing import Bin
from analysis.util import get_unique_matrix
from experiment import ExpContext, TunerContext
from globals import Paths
from common.timeutil import stopwatch

class WorkloadState(object):

    def __init__(self, X=None, y=None, models=None):
        self.X = X
        self.y = y
        self.models = models

    @staticmethod
    def compress(workload_state):
        return zlib.compress(pickle.dumps(workload_state))

    @staticmethod
    def decompress(compressed_workload_state):
        return pickle.loads(zlib.decompress(compressed_workload_state))

def worker_create_model((workload_name, data)):
    X, y = data
    models = {}
    for col, label in zip(y.data.T, y.columnlabels):
        print "building model: {}".format(label)
        length_scale, magnitude, ridge_const = 1., 1., 1.
        ridge = np.ones(X.data.shape[0]) * ridge_const
        col = col.reshape(-1, 1)
        model = GPR(length_scale, magnitude)
        model.fit(X.data, col, ridge)
        models[label] = model.get_params()
    workload_state = WorkloadState(X, y, models)
    print "Done.\n"
    return (workload_name, workload_state)

def worker_score_workload((workload_name, workload_state, X_client, y_client)):
    assert np.array_equal(workload_state.X.columnlabels,
                          X_client.columnlabels)
    assert np.array_equal(workload_state.y.columnlabels,
                          y_client.columnlabels)

    # Make all predictions
    model = GPR()
    model._reset()
    metrics = workload_state.y.columnlabels
    predictions = np.empty_like(y_client.data)
    for i, metric in enumerate(metrics):
        model_params = workload_state.models[metric]
        model.set_params(**model_params)
        res = model.predict(X_client.data)
        predictions[:, i] = res.ypreds.ravel()

#     print "predictions={}".format(predictions)
#     print ""
#     print "client={}".format(y_client.data)
#     print ""

    # Compute distance
    dists = np.sum(np.square(np.subtract(predictions, y_client.data)), axis=1)
#     print "dists={}".format(dists)
    assert dists.shape == (predictions.shape[0],)
#     print ""
#     print "score={}".format(np.mean(dists))
    return (workload_name, np.mean(dists))

class WorkloadMapper(object):

    NUM_KNOBS = 12
    MAX_METRICS = 5
    POOL_SIZE = 2
    
    def __init__(self):
        exp = ExpContext()
        
        dbms = exp.dbms.name
        cluster = exp.server.instance_type
        workload_dirs = glob.glob(os.path.join(Paths.DATADIR,
                                               "analysis*{}*{}*".format(dbms,
                                                                        cluster)))
        assert len(workload_dirs) > 0
        for w in workload_dirs:
            assert "session" not in w

        self.workload_dirs_ = workload_dirs
        self.initialize_models(self.NUM_KNOBS)
        gc.collect()

    def initialize_models(self, n_knobs):
        tuner = TunerContext()

        with stopwatch("model initialization"):
            featured_knobs = tuner.get_n_benchmark_featured_knobs(n_knobs)
            featured_metrics = tuner.featured_metrics[:self.MAX_METRICS]
            self.X_scaler_ = StandardScaler()
            self.y_scaler_ = StandardScaler()
            data_map = {}

            # Load data and fit scalers
            for wd in self.workload_dirs_:
                Xpath = os.path.join(wd, "X_data_enc.npz")
                ypath = os.path.join(wd, "y_data_enc.npz")
                X = Matrix.load_matrix(Xpath)
                y = Matrix.load_matrix(ypath)
                X = X.filter(featured_knobs, "columns")
                y = y.filter(featured_metrics, "columns")
                assert np.array_equal(X.columnlabels, featured_knobs)
                assert np.array_equal(y.columnlabels, featured_metrics)
                
                self.X_scaler_.partial_fit(X.data)
                self.y_scaler_.partial_fit(y.data)
                data_map[wd] = (X, y)

            # Scale X/y
            all_ys = []
            for wd, (X, y) in data_map.iteritems():
                X.data = self.X_scaler_.transform(X.data)
                y.data = self.y_scaler_.transform(y.data)
                all_ys.append(y.data)

            # Concat all ys and compute deciles
            all_ys = np.vstack(all_ys)
            self.y_binner_ = Bin(0, axis=0)
            self.y_binner_.fit(all_ys)
            self.y_gp_scaler_ = StandardScaler()
            self.y_gp_scaler_.fit(all_ys)
            del all_ys

            # Bin y by deciles and recenter
            for wd, (X, y) in data_map.iteritems():
                y.data = self.y_binner_.transform(y.data)
                y.data = self.y_gp_scaler_.transform(y.data)
            
            p = multiprocessing.Pool(self.POOL_SIZE)
            res = p.map(worker_create_model, list(data_map.iteritems())[:2])
            self.workload_states_ = dict(res)

            test_wkld_name, test_wkld = list(self.workload_states_.iteritems())[0]
            indices = np.random.choice(np.arange(test_wkld.X.data.shape[0]), 5)
            X_client = Matrix(test_wkld.X.data[indices],
                              test_wkld.X.rowlabels[indices],
                              test_wkld.X.columnlabels)
            y_client = Matrix(test_wkld.y.data[indices],
                              test_wkld.y.rowlabels[indices],
                              test_wkld.y.columnlabels)
            iterable = [(wd, ws, X_client, y_client) for \
                        wd, ws in self.workload_states_.iteritems()]
            p = multiprocessing.Pool(self.POOL_SIZE)
            res = p.map(worker_score_workload, iterable)
            print "TEST_WKLD_NAME: {}\n".format(test_wkld_name)
            for wkld_name, score in res:
                print "{}: {}".format(wkld_name, score)


    def map_workload(self, X_client, y_client):
        tuner = TunerContext()
 
        # Filter be featured knobs & metrics
        X_client = X_client.filter(tuner.featured_knobs, "columns")
        y_client = y_client.filter(tuner.featured_metrics, "columns")
         
        # Generate unique X,y matrices
        X_client, y_client = get_unique_matrix(X_client, y_client)
        
        # Preprocessing steps
        X_client.data = self.X_scaler_.transform(X_client.data)
        
        # TODO: fixme!
        y_client.data = self.y_scaler_.transform(y_client.data)
        y_client.data = self.y_binner_.transform(y_client.data)
        y_client.data = self.y_gp_scaler_.transform(y_client.data)
        
        model = GPR()
        score_map = {}
        for wd, workload_state in self.workload_states_.iteritems():
            workload_state = WorkloadState.decompress(workload_state)
            assert np.array_equal(workload_state.X.columnlabels,
                                  X_client.columnlabels)
            assert np.array_equal(workload_state.y.columnlabels,
                                  y_client.columnlabels)
            
            # Predict each metric
            predictions = np.empty_like(y_client.data)
            for i, metric in enumerate(workload_state.y.columnlabels):
                model_params = workload_state.models[metric]
                model.set_params(**model_params)
                gpres = model.predict(X_client.data)
                predictions[i] = gpres.ypreds

            # Compute distance
            dists = np.sum(np.square(np.subtract(predictions, y_client.data)), axis=1)
            print dists.shape
            assert dists.shape == (predictions.shape[0], 1)
            score_map[wd] = np.mean(dists)
        
#         
#         # Bin the metrics using the precomputed deciles
#         binned_y = self.binner_.transform(y_client.data)
# 
#         wkld_scores = {}
#         if tuner.incremental_knob_selection and tuner.num_knobs != self.tuned_for_:
#             self.prep_matrices()
#         #with matlab_engine() as engine:
#         for wkld,(y_mtx, X_data) in self.matrices_.iteritems():
#             wkld_score = 0.0
#             row_indices = [(ci,i) for i,l in enumerate(y_mtx.rowlabels) \
#                            for ci,cl in enumerate(y_client.rowlabels) \
#                            if l == cl]
#             missing_indices = [i for i in range(len(y_client.rowlabels)) \
#                                if i not in row_indices]
#             if len(row_indices) > 0:
#                 wkld_score += np.sum([euclidean(binned_y[ci], y_mtx.data[i]) \
#                                       for ci,i in row_indices])
#             
#             if len(missing_indices) > 0:
#                 # Predict any missing indices
#                 X_test = X_client.data[missing_indices]
#                 ridge = 0.000001 * np.ones(X_data.shape[0])
#                 predictions = []
#                 for ycol in y_mtx.data.T:
#                     # Make predictions
#                     ypreds, _, _ = tf_predict(X_data, ycol.reshape(ycol.shape[0],1), X_test, ridge)
#                     ypreds = np.array(np.array(ypreds, dtype=int), dtype=float)
#                     ypreds[ypreds > 9] = 9
#                     ypreds[ypreds < 0] = 0
#                     predictions.append(ypreds.ravel())
#                 
#                 # Create new matrix out of predictions
#                 ypreds = np.vstack(predictions).T
#                 ypreds = Matrix(ypreds, y_client.rowlabels[missing_indices],
#                                 y_client.columnlabels)
# 
#                 # Update y matrix and X data with new predictions
#                 new_y_mtx = Matrix.vstack([y_mtx, ypreds], require_equal_columnlabels=True)
#                 new_X_data = np.vstack([X_data, X_test])
#                 self.matrices_[wkld] = (new_y_mtx, new_X_data)
#                 
#                 # Update score using new predictions
#                 wkld_score += np.sum([euclidean(u, v) for u,v in \
#                                       zip(binned_y[missing_indices],
#                                           ypreds.data)])
#                 gc.collect()
#             wkld_scores[wkld] = wkld_score / X_client.data.shape[0]
# 
#         winners = sorted(wkld_scores.items(), key=operator.itemgetter(1))
#         print "\nWINNERs:\n{}\n".format(winners)
#         
#         if tuner.map_to == "best":
#             if ExpContext().benchmark.bench_name == "tpch":
#                 winner = [w for w in self.matrices_.keys() if "tpch" in w]
#                 assert len(winner) == 1
#                 winner = winner[0]
#             else:
#                 winner = winners[0][0]
#         else:
#             winner = winners[-1][0]
#         
#         print "Mapping to {}\n\n".format(winner)
#         return winner
