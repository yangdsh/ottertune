'''
Created on Jul 11, 2016

@author: dvanaken
'''
import os.path
import gc, copy
import glob, zlib
import dill as pickle
import multiprocessing
import numpy as np
import operator
from sklearn.preprocessing import StandardScaler

from .gp_tf import GPR
from .matrix import Matrix
import analysis.preprocessing as prep
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

def worker_create_model((worker_id, workload_name, data, njobs, verbose)):
    if verbose:
        print "{}: building models for {}".format(worker_id,
                                                  os.path.basename(workload_name))
    X, y = data
    models = {}
    for col, label in zip(y.data.T, y.columnlabels):
        if verbose:
            print "    {}: {}".format(worker_id, label)
        length_scale, magnitude, ridge_const = 1., 1., 1.
        ridge = np.ones(X.data.shape[0]) * ridge_const
        col = col.reshape(-1, 1)
        model = GPR(length_scale, magnitude)
        model.fit(X.data, col, ridge)
        models[label] = model.get_params()
    workload_state = WorkloadState(X, y, models)
    if verbose:
        print "{}: done. ({}/{})".format(worker_id, worker_id+1, njobs)
    return (workload_name, workload_state)

def worker_score_workload((worker_id, workload_name, workload_state,
                           X_client, y_client, njobs, verbose)):
    if verbose:
        print "{}: computing scores for {}".format(worker_id,
                                                   os.path.basename(workload_name))
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
        if verbose:
            print "    {}: {}".format(worker_id, metric)
        model_params = workload_state.models[metric]
        model.set_params(**model_params)
        res = model.predict(X_client.data)
        predictions[:, i] = res.ypreds.ravel()

    # Compute distance
    dists = np.sum(np.square(np.subtract(predictions, y_client.data)), axis=1)
    assert dists.shape == (predictions.shape[0],)
    if verbose:
        print "{}: done. ({}/{})".format(worker_id, worker_id+1, njobs)
    return (workload_name, np.mean(dists))

class WorkloadMapper(object):

    POOL_SIZE = 2
    
    def __init__(self, verbose=True):
        exp = ExpContext()
        tuner = TunerContext()
        self.verbose_ = verbose
        self.featured_knobs_ = tuner.featured_knobs
        self.workload_states_ = None
        
        dbms = exp.dbms.name
        cluster = exp.server.instance_type
        workload_dirs = glob.glob(os.path.join(Paths.DATADIR,
                                               "analysis*{}*{}*".format(dbms,
                                                                        cluster)))
        assert len(workload_dirs) > 0
        target_wkld_desc = exp.exp_id(exp.benchmark)
        self.workload_dirs_ = [w for w in workload_dirs if not \
                               w.endswith(target_wkld_desc)]
        self.initialize_models()
        assert self.workload_states_ is not None
        gc.collect()

    def initialize_models(self):
        exp = ExpContext()
        tuner = TunerContext()

        if self.verbose_:
            print ("Initializing models for # knobs={}\n"
                   .format(self.featured_knobs_.size))
        with stopwatch("workload mapping model creation"):
            n_values, cat_indices, params = prep.dummy_encoder_helper(exp.dbms.name,
                                                                      self.featured_knobs_)
            if n_values.size > 0:
                self.dummy_encoder_ = prep.DummyEncoder(n_values, cat_indices)
            else:
                self.dummy_encoder_ = None
            self.X_scaler_ = StandardScaler()
            self.y_scaler_ = StandardScaler()
            data_map = {}
            for i,wd in enumerate(self.workload_dirs_):
                # Load and filter data
                Xpath = os.path.join(wd, "X_data_enc.npz")
                ypath = os.path.join(wd, "y_data_enc.npz")
                X = Matrix.load_matrix(Xpath)
                y = Matrix.load_matrix(ypath)
                X = X.filter(self.featured_knobs_, "columns")
                y = y.filter(tuner.featured_metrics, "columns")
                assert np.array_equal(X.columnlabels, self.featured_knobs_)
                assert np.array_equal(y.columnlabels, tuner.featured_metrics)
                assert np.array_equal(X.rowlabels, y.rowlabels)
                
                # Dummy-code categorical knobs
                if self.dummy_encoder_ is not None:
                    if i == 0:
                        # Just need to fit this once
                        self.dummy_encoder_.fit(X.data, columnlabels=X.columnlabels)
                    X = Matrix(self.dummy_encoder_.transform(X.data),
                               X.rowlabels,
                               self.dummy_encoder_.columnlabels)
                
                self.X_scaler_.partial_fit(X.data)
                self.y_scaler_.partial_fit(y.data)
                data_map[wd] = (X, y)
            
            if self.dummy_encoder_ is not None:
                # Fix X_scaler wrt categorical features
                prep.fix_scaler(self.X_scaler_, self.dummy_encoder_, params)

            # Scale X/y
            all_ys = []
            for wd, (X, y) in data_map.iteritems():
                X.data = self.X_scaler_.transform(X.data)
                y.data = self.y_scaler_.transform(y.data)
                all_ys.append(y.data)

            # Concat all ys and compute deciles
            all_ys = np.vstack(all_ys)
            self.y_binner_ = prep.Bin(0, axis=0)
            self.y_binner_.fit(all_ys)
            self.y_gp_scaler_ = StandardScaler()
            self.y_gp_scaler_.fit(all_ys)

            # Bin y by deciles and recenter
            for wd, (X, y) in data_map.iteritems():
                y.data = self.y_binner_.transform(y.data)
                y.data = self.y_gp_scaler_.transform(y.data)
            
            njobs = len(data_map)
            iterable = [(i,wd,ws,njobs,self.verbose_) for i,(wd,ws) \
                        in enumerate(data_map.iteritems())]            
            p = multiprocessing.Pool(self.POOL_SIZE)
            res = p.map(worker_create_model, iterable)
            p.terminate()
            self.workload_states_ = dict(res)

    def map_workload(self, X_client, y_client):
        tuner = TunerContext()

        # Recompute the GPR models if the # of knobs to tune has
        # changed (incremental knob selection feature is enabled)
        tuner_feat_knobs = tuner.featured_knobs
        if not np.array_equal(tuner_feat_knobs, self.featured_knobs_):
            assert tuner_feat_knobs.size != self.featured_knobs_.size
            assert tuner.incremental_knob_selection == True
            self.featured_knobs_ = tuner_feat_knobs
            self.initialize_models()
            gc.collect()

        # Filter be featured knobs & metrics
        X_client = X_client.filter(self.featured_knobs_, "columns")
        y_client = y_client.filter(tuner.featured_metrics, "columns")
         
        # Generate unique X,y matrices
        X_client, y_client = get_unique_matrix(X_client, y_client)
        
        # Preprocessing steps
        if self.dummy_encoder_ is not None:
            X_client = Matrix(self.dummy_encoder_.transform(X_client.data),
                              X_client.rowlabels,
                              self.dummy_encoder_.columnlabels)
        X_client.data = self.X_scaler_.transform(X_client.data)
        
        # Create y_client scaler with prior and transform client data
        y_client_scaler = copy.deepcopy(self.y_scaler_)
        y_client_scaler.n_samples_seen_ = 5
        y_client_scaler.partial_fit(y_client.data)
        y_client.data = self.y_client_scaler.transform(y_client.data)
        
        # Bin and recenter client data
        y_client.data = self.y_binner_.transform(y_client.data)
        y_client.data = self.y_gp_scaler_.transform(y_client.data)

        # Compute workload scores in parallel
        njobs = len(self.workload_states_)
        iterable = [(i, wd, ws, X_client, y_client, njobs, self.verbose_) \
                    for i,(wd,ws) in self.workload_states_.iteritems()]

        p = multiprocessing.Pool(self.POOL_SIZE)
        wkld_scores = p.map(worker_score_workload, iterable)
        p.terminate()

        sorted_wkld_scores = sorted(wkld_scores, key=operator.itemgetter(1))
        if tuner.map_to == "worst":
            sorted_wkld_scores = sorted_wkld_scores[::-1]
        else:
            assert tuner.map == "best"
        
        return sorted_wkld_scores[0][0]
