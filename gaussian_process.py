'''
Created on Jul 11, 2016

@author: dvanaken
'''

import os.path, copy
import numpy as np

from common.timeutil import stopwatch

JITTER = 1e-6
N_LOCAL_POINTS = 20
N_GLOBAL_POINTS = 200

def get_next_config(X_client, y_client, workload_name=None):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    from .constraints import ParamConstraintHelper
    from .gp_tf import GPR_GD
    from .matrix import Matrix
    from .util import get_exp_labels, get_unique_matrix
    import analysis.preprocessing as prep
    from benchmark.early_abort import EarlyAbortConfig
    from experiment import ExpContext, TunerContext
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
    assert np.array_equal(X_client.columnlabels, featured_knobs)
    y_client = y_client.filter(np.array([tuner.optimization_metric]), "columns")
    last_exp_idx = [i for i,e in enumerate(y_client.rowlabels) \
                    if os.path.basename(e) == tuner.get_last_exp()]
    assert len(last_exp_idx) == 1
    tuner.append_opt_metric(np.asscalar(y_client.data[last_exp_idx]))
    
    # Get median latencies for early abort config
    if exp.benchmark.is_olap:
        latencies_us = get_query_response_times()
    else:
        latencies_us = int(y_client.data.min() * 1000)

    # Update client rowlabels
    X_client.rowlabels = get_exp_labels(X_client.data, X_client.columnlabels)
    y_client.rowlabels = X_client.rowlabels.copy()
    print X_client
    print y_client
    num_observations = X_client.data.shape[0] 
    n_values, cat_knob_indices, params = prep.dummy_encoder_helper(exp.dbms.name,
                                                                   featured_knobs)
    if n_values.size > 0:
        encoder = prep.DummyEncoder(n_values, cat_knob_indices)
    else:
        encoder = None

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
            
            # Filter out all metrics except for the optimization metric
            y_train = Matrix.load_matrix(y_path)
            y_train = y_train.filter(np.array([tuner.optimization_metric]), "columns")
            assert np.array_equal(y_train.columnlabels, y_client.columnlabels)

            if tuner.unique_training_data:
                # Get X,y matrices with unique samples
                X_train, y_train = get_unique_matrix(X_train, y_train)
            else:
                # Use all samples
                X_train.rowlabels = get_exp_labels(X_train.data, X_train.columnlabels)
                y_train.rowlabels = X_train.rowlabels.copy()
            
            assert np.array_equal(X_train.columnlabels, X_client.columnlabels)
            
            # Create y train scaler
            y_train_scaler = StandardScaler()
            y_train_scaler.fit(y_train.data)
            y_train.data = y_train_scaler.transform(y_train.data)
            
            y_client_scaler = copy.deepcopy(y_train_scaler)
            y_client_scaler.n_samples_seen_ = 2
            y_client_scaler.partial_fit(y_client.data)
            y_client.data = y_client_scaler.transform(y_client.data)
            y_scaler = y_client_scaler
            
            # Concatenate workload and client matrices to create X_train/y_train
            client_data_idxs = []
            for cidx, rowlabel in enumerate(X_client.rowlabels):
                primary_idxs = np.array([idx for idx,rl in enumerate(X_train.rowlabels) \
                                if rl == rowlabel])
                if len(primary_idxs) >= 1:
                    # Replace client results in workload matrix if overlap
                    y_train.data[primary_idxs] = y_client.data[cidx]
                    client_data_idxs.append(primary_idxs)
    
            client_data_idxs.append(np.arange(X_client.data.shape[0]) + X_train.data.shape[0])
            client_data_idxs = np.hstack(client_data_idxs)
            X_train = Matrix.vstack([X_train, X_client])
            y_train = Matrix.vstack([y_train, y_client])
        else:
            y_scaler = StandardScaler()
            y_client.data = y_scaler.fit_transform(y_client.data)
            
            X_train = X_client
            y_train = y_client
            client_data_idxs = np.arange(X_train.data.shape[0])
        hps = get_hyperparameters(client_data_idxs,
                                  X_train.data.shape[0],
                                  workload_name)

        # Encode/scale X_train
        if encoder is not None:
            encoder.fit(X_client.data, columnlabels=featured_knobs)
            X_train = Matrix(encoder.transform(X_train.data),
                             X_train.rowlabels,
                             encoder.columnlabels)
    tuner.append_stat("gp_preprocessing_sec", t.elapsed_seconds)
    
    with stopwatch() as t:
        n_local_points, n_global_points = N_LOCAL_POINTS, N_GLOBAL_POINTS
        if X_train.data.shape[0] < n_local_points:
            n_local_points = X_train.data.shape[0]
        search_data = np.empty((n_local_points + n_global_points,
                                X_train.data.shape[1]))

        X_scaler = StandardScaler()
        X_scaler.partial_fit(X_train.data)

        # Generate global search points
        config_mgr = exp.dbms.config_manager_
        if n_global_points > 0:
            X_test_data = config_mgr.get_param_grid(featured_knobs)
            X_test_data = X_test_data[np.random.choice(np.arange(X_test_data.shape[0]),
                                                   n_global_points,
                                                   replace=False)]
        else:
            X_test_data = None

        #if n_local_points > 0: # TODO: remove after debug
        #    print ""
        #    print "---------------------------------------"
        #    print "LOCAL POINTS:"
        #    best_indices = np.argsort(y_train.data.ravel())[:n_local_points]
        #    for row in X_train.rowlabels[best_indices]:
        #        print row
        #    print "---------------------------------------"
        #    print ""
        if X_test_data is not None:
            if encoder is not None:
                X_test_data = encoder.transform(X_test_data)
        #mins, maxs = prep.get_min_max(params, encoder)
        #X_scaler = prep.MinMaxScaler(mins=mins, maxs=maxs)
            X_scaler.partial_fit(X_test_data)
        if encoder is not None:
            prep.fix_scaler(X_scaler, encoder, params)
        X_train.data = X_scaler.transform(X_train.data)
        if X_test_data is not None:
            X_test_data = X_scaler.transform(X_test_data)
            search_data[:n_global_points,:] = X_test_data

        # Find local search points
        if n_local_points > 0:
            best_indices = np.argsort(y_train.data.ravel())[:n_local_points]
            search_data[n_global_points:] = X_train.data[best_indices] + JITTER
    tuner.append_stat("gpr_search_points_sec", t.elapsed_seconds)
    
    with stopwatch() as t:
        # Run GPR/GD
        constraint_helper = ParamConstraintHelper(params, X_scaler, encoder)
        sigma_multiplier = GPR_GD.calculate_sigma_multiplier(t=num_observations,
                                                             ndim=X_train.data.shape[1])
        #sigma_multiplier = 3.0
        tuner.append_stat("gpr_sigma_multiplier", sigma_multiplier)
        print "SIGMA MULTIPLIER: {0:.2f}".format(sigma_multiplier)
        gpr = GPR_GD(length_scale=hps['length_scale'],
                     magnitude=hps['magnitude'],
                     sigma_multiplier=sigma_multiplier)
        gpr.fit(X_train.data, y_train.data, hps['ridge'])
        gpres = gpr.predict(search_data, constraint_helper)
    tuner.append_stat("gpr_compute_time_sec", t.elapsed_seconds)

    #for i in range(n_local_points + n_global_points):
    #    print "{}. y={}, sigma={}, minL={}, conf={}".format(i+1, gpres.ypreds[i], gpres.sigmas[i], gpres.minL[i], constraint_helper.get_valid_config(gpres.minL_conf[i], rescale=False))

    # Replace categorical features with original
    #print "ypreds: {}".format(gpres.ypreds)
    #print "sigmas: {}".format(gpres.sigmas)
    #print "minL: {}".format(gpres.minL)
    #print "minL_conf: {}".format(gpres.minL_conf.shape)
    print ""
    next_config_idx = np.nanargmin(gpres.minL.ravel())
    print "next_config_idx: {}, {}".format(next_config_idx, next_config_idx.shape)
    next_config = gpres.minL_conf[next_config_idx]
    print "next_conf #1: {}".format(next_config)
    next_config = constraint_helper.get_valid_config(next_config, rescale=False)
    print "next_conf #2: {}".format(next_config)
    tuner.append_gp_info(gpres.ypreds[next_config_idx].tolist(),
                         gpres.sigmas[next_config_idx].tolist(),
                         y_scaler.__dict__)
    
    # Decode config into actual config parameters
    param_pairs = [(k, v) for k,v in zip(featured_knobs, next_config)]
    next_config_params = config_mgr.decode_params(param_pairs)
    tuner.append_dbms_config(next_config_params)
    for k,v in sorted(next_config_params.iteritems()):
        print "{}: {}".format(k,v)
    print ""
    dbms_config = config_mgr.get_next_config(next_config_params)
    #abort_config = EarlyAbortConfig.create_config(latencies_us,abort_threshold_percentage=50)
    abort_config = EarlyAbortConfig.get_default_config()
    return dbms_config, abort_config

def get_query_response_times():
    import json
    from common.fileutil import FileExtensions, ResultFiler
    from experiment import ExpContext
    
    exp = ExpContext()
    num_queries = np.sum(exp.benchmark.work_items[0].weights)
    
    raw_files = ResultFiler.get_all_result_files(exp.benchmark,
                                                 FileExtensions.RAW,
                                                 tuning_session=True)
    assert len(raw_files) > 0
    
    exec_times = np.empty((len(raw_files), num_queries))
    for i,rf in enumerate(raw_files):
        queries = json.loads(rf)['samples']
        assert len(queries) == num_queries
        for j,query in enumerate(queries):
            exec_times[i,j] = float(query[2])
    median_exec_times = np.median(exec_times, axis=0)
    assert median_exec_times.size == num_queries

    # Convert from ms to us
    median_exec_times *= 1000
    
    return median_exec_times
    
def get_hyperparameters(client_indices, ntrain, workload_name=None):
    hyperparams = {}
    hyperparams['length_scale'] = 1.0
    hyperparams['magnitude'] = 1.5
    hyperparams['ridge'] = np.ones((ntrain,)) * 5.0
    hyperparams['ridge'][client_indices] = 1.0
    return hyperparams

