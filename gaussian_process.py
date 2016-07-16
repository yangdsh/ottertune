'''
Created on Jul 11, 2016

@author: dvanaken
'''

import os.path, gc
import numpy as np

from common.timeutil import stopwatch

def get_next_config(X_client, y_client, workload_name=None):
    from sklearn.preprocessing import StandardScaler
    from analysis.matrix import Matrix
    from analysis.util import get_exp_labels, get_unique_matrix
    from experiment import ExpContext, TunerContext, matlab_engine
    from globals import Paths
    
    exp = ExpContext()
    tuner = TunerContext()
    
    # Choose the featured knobs per benchmark experiments or
    # the featured knobs per DBMS experiments (tuner setting)
    if tuner.gp_featured_knobs_scope == "benchmark":
        featured_knobs = tuner.benchmark_featured_knobs
    else:
        featured_knobs = tuner.featured_knobs
    print ""
    print "featured knobs ({}):".format(tuner.gp_featured_knobs_scope)
    print featured_knobs
    print ""
    
    # Filter the featured knobs & metrics
    X_client = X_client.filter(featured_knobs, "columns")
    y_client = y_client.filter(np.array([tuner.optimization_metric]), "columns")
    last_exp_idx = [i for i,e in enumerate(y_client.rowlabels) \
                    if os.path.basename(e) == tuner.get_last_exp()]
    assert len(last_exp_idx) == 1
    tuner.append_opt_metric(y_client.data[last_exp_idx])

    # Update client rowlabels
    X_client.rowlabels = get_exp_labels(X_client.data, X_client.columnlabels)
    y_client.rowlabels = X_client.rowlabels.copy()
    print X_client
    print y_client

    with stopwatch() as t:
        if tuner.map_workload:
            assert workload_name is not None

            # Load data for mapped workload
            datadir = os.path.join(Paths.DATADIR, workload_name)
            X_path = os.path.join(datadir, "X_data_enc.npz".format(tuner.num_knobs))
            y_path = os.path.join(datadir, "y_data_enc.npz".format(tuner.num_knobs))
            
            # Load X and filter out non-featured knobs
            X_train = Matrix.load_matrix(X_path).filter(featured_knobs, "columns")
            assert np.array_equal(X_train.columnlabels, featured_knobs)
            assert np.array_equal(X_train.columnlabels, X_client.columnlabels)
            
            # Filter out all metrics except for the optimization metric
            y_train = Matrix.load_matrix(y_path)
            y_train = y_train.filter(np.array([tuner.optimization_metric]), "columns")
            assert np.array_equal(y_train.columnlabels, y_client.columnlabels)
            
            # Get X,y matrices with unique rows
            if tuner.unique_training_data:
                X_train, y_train = get_unique_matrix(X_train, y_train)
            else:
                X_train.rowlabels = get_exp_labels(X_train.data, X_train.columnlabels)
                y_train.rowlabels = X_train.rowlabels.copy()
    
            print ""
            print "orig X_train shape: {}".format(X_train.data.shape)
            print "orig y_train shape: {}".format(y_train.data.shape)
            print ""
            
            # Concatenate workload and client matrices to create X_train/y_train
            ridge = 0.01 * np.ones(X_train.data.shape[0])
            for cidx, rowlabel in enumerate(X_client.rowlabels):
                primary_idxs = [idx for idx,rl in enumerate(X_train.rowlabels) \
                            if rl == rowlabel]
                #assert len(primary_idx) <= 1
                if len(primary_idxs) == 1:
                    # Replace client results in workload matrix if overlap
                    y_train.data[primary_idxs] = y_client.data[cidx]
                    ridge[primary_idxs] = 0.000001
    
            X_train = Matrix.vstack([X_train, X_client])
            y_train = Matrix.vstack([y_train, y_client])
            ridge = np.append(ridge, 0.000001 * np.ones(X_client.data.shape[0]))
        else:
            X_train = X_client
            y_train = y_client
            ridge = 0.000001 * np.ones(X_train.data.shape[0])
    tuner.append_stat("gp_preprocessing_sec", t.elapsed_seconds)
    
    with stopwatch() as t:
        # Generate grid to create X_test
        config_mgr = exp.dbms.config_manager_
        X_test = config_mgr.get_param_grid(featured_knobs)
        X_test_rowlabels = get_exp_labels(X_test, featured_knobs)
        X_test = Matrix(X_test, X_test_rowlabels, featured_knobs)
    tuner.append_stat("create_param_grid_sec", t.elapsed_seconds)
        
    # Scale X_train, y_train and X_test
    X_standardizer = StandardScaler()
    y_standardizer = StandardScaler()
    X_train.data = X_standardizer.fit_transform(X_train.data)
    X_test.data = X_standardizer.fit_transform(X_test.data)
    y_train.data = y_standardizer.fit_transform(y_train.data)

    print "X_train shape: {}".format(X_train.data.shape)
    print "y_train shape: {}".format(y_train.data.shape)
    print "X_test shape: {}".format(X_test.data.shape)
    
    with stopwatch() as t:
        if tuner.gp_algorithm == "matlab":

            # Make predictions
            with matlab_engine() as engine:
                ypreds, sigmas, eips = predict(X_train.data,
                                               y_train.data,
                                               X_test.data,
                                               ridge,
                                               engine)
        
        elif tuner.gp_algorithm == "sklearn":
            assert tuner.config_selection_mode == "sigma"
            ypreds, sigmas, eips = sklearn_predict(X_train.data,
                                                   y_train.data,
                                                   X_test.data,
                                                   ridge)
        else:
            raise Exception("Unknown GP version: {}".format(tuner.gp_algorithm))
    tuner.append_stat("gpr_compute_time_sec", t.elapsed_seconds)
    gc.collect()

    ypreds_unscaled = y_standardizer.inverse_transform(ypreds)

    print "best: delta -"
    sigma_idx = get_best_idx(ypreds, sigmas, eips, method="sigma")
    debug_idx(X_test, ypreds_unscaled, ypreds, sigmas, eips, sigma_idx)
    print ""
    print "best: eip -"
    eip_idx = get_best_idx(ypreds, sigmas, eips, method="eip")
    debug_idx(X_test, ypreds_unscaled, ypreds, sigmas, eips, eip_idx)
    print ""
    
    if tuner.config_selection_mode == "sigma":
        selection_idx = sigma_idx
    elif tuner.config_selection_mode == "eip":
        selection_idx = eip_idx
    else:
        raise Exception("Unknown config selection mode: {}"
                        .format(tuner.config_selection_mode))
    
    # Log 'winning' config info
    tuner.append_gp_info(ypreds[selection_idx],
                         sigmas[selection_idx],
                         eips[selection_idx],
                         y_standardizer.__dict__)

    # Config manager must decode any categorical parameters
    next_config_params = config_mgr.decode_params(X_test.rowlabels[selection_idx])
    for pname, pval in next_config_params.iteritems():
        print pname,":",pval

    tuner.append_dbms_config(next_config_params)
    config = config_mgr.get_next_config(next_config_params)
    return config

def get_best_idx(ypreds, sigmas, eips, method="sigma"):
    if method == "sigma":
        best_idx = np.argmin(ypreds - sigmas)
    elif method == "eip":
        best_idx = np.argmax(eips)
    else:
        raise Exception("Unknown method: {}".format(method))
    return best_idx

def debug_idx(xs, ys, ypreds, sigmas, eips, idx):
    print ""
    print "y_real={}, y_scaled={}, sigma={}, eip={}".format(ys[idx],
                                                            ypreds[idx],
                                                            sigmas[idx],
                                                            eips[idx])
    print xs.rowlabels[idx]
    print ""

def sklearn_predict(X_train, y_train, X_test, ridge, batch_size=1000):
    import multiprocessing
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel
    
    gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0, (1e-2, 1e2)),
                                  alpha=ridge)

    gp.fit(X_train, y_train)

    sections = X_test.shape[0] / batch_size
    X_test_chunks = np.vsplit(X_test, sections)
    pool_size = 5
    p = multiprocessing.Pool(pool_size)
    iterable = [(i,chunk,gp) for i,chunk in enumerate(X_test_chunks)]

    res = p.map(sklearn_predict_helper, iterable)
    ypreds = [r[0] for r in res]
    sigmas = [r[1] for r in res]
    ypreds = np.hstack(ypreds)
    sigmas = np.hstack(sigmas)
    
    return ypreds, sigmas, np.zeros(len(ypreds))

def sklearn_predict_helper((chunk_idx, chunk, gp)):
    print "Starting chunk #{}".format(chunk_idx)
    with stopwatch("chunk #{}".format(chunk_idx)):
        y,s = gp.predict(chunk, return_std=True)
    return y.ravel(), s

def predict(X_train, y_train, X_test, ridge, eng):
    n_feats = X_train.shape[1]
    X_train = X_train.ravel()
    y_train = y_train.ravel()
    X_test = X_test.ravel()
    ridge = ridge.ravel()

    X_train = X_train.squeeze() if X_train.ndim > 1 else X_train
    y_train = y_train.squeeze() if y_train.ndim > 1 else y_train
    X_test = X_test.squeeze() if X_test.ndim > 1 else X_test
    ridge = ridge.squeeze() if ridge.ndim > 1 else ridge

    ypreds, sigmas, eips = eng.gp(X_train.tolist(),
                                    y_train.tolist(),
                                    X_test.tolist(),
                                    ridge.tolist(),
                                    n_feats,
                                    nargout=3)
    ypreds = [ypreds] if isinstance(ypreds, float) else list(ypreds)
    sigmas = [sigmas] if isinstance(sigmas, float) else list(sigmas)
    ypreds = [eips] if isinstance(eips, float) else list(eips)

    return np.array(ypreds, dtype=float), \
           np.array(sigmas, dtype=float), \
           np.array(eips, dtype=float)

