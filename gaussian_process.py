'''
Created on Jul 11, 2016

@author: dvanaken
'''

import os.path
import numpy as np

def get_next_config(workload_name, X_client, y_client):
    
    from analysis.matrix import Matrix
    from analysis.preprocessing import Standardize
    from analysis.util import get_unique_matrix, get_exp_labels
    from experiment import ExpContext, TunerContext
    from globals import Paths
    
    exp = ExpContext()
    tuner = TunerContext()
    
    # Load data for mapped workload
    datadir = os.path.join(Paths.DATADIR, workload_name)
    X_path = os.path.join(datadir, "X_data_unique_{}.npz".format(tuner.num_knobs))
    y_path = os.path.join(datadir, "y_data_unique_{}.npz".format(tuner.num_knobs))
    X_train = Matrix.load_matrix(X_path)
    assert np.array_equal(X_train.columnlabels, tuner.featured_knobs)
    assert np.array_equal(X_train.columnlabels, X_client.columnlabels)
    
    # Filter out all metrics except for the optimization metric
    y_train = Matrix.load_matrix(y_path)
    y_train = y_train.filter(np.array([tuner.optimization_metric]), "columns")
    y_client = y_client.filter(np.array([tuner.optimization_metric]), "columns")
    assert np.array_equal(y_train.columnlabels, y_client.columnlabels)
    
    # Convert client matrices into same format as wkld matrices
    X_client, y_client = get_unique_matrix(X_client, y_client)

    # Concatenate workload and client matrices to create X_train/y_train
    ridge = 0.01 * np.ones((X_train.shape[0],))
    new_result_idxs = []
    for cidx, rowlabel in enumerate(X_client.rowlabels):
        primary_idx = [idx for idx,rl in enumerate(X_train.rowlabels) \
                       if rl == rowlabel]
        assert len(primary_idx) <= 1
        if len(primary_idx) == 1:
            # Replace client results in workload matrix if overlap
            y_train.data[primary_idx] = y_client.data[cidx]
            ridge[primary_idx] = 0.000001
        else:
            # Else this is a unique client result
            new_result_idxs.append(cidx)
    
    X_train = Matrix.vstack([X_train, X_client[new_result_idxs]])
    y_train = Matrix.vstack([y_train, y_client[new_result_idxs]])
    ridge[-len(new_result_idxs):] = 0.000001
    
    # Generate grid to create X_test
    config_mgr = exp.dbms.config_manager_
    X_test = config_mgr.get_param_grid(tuner.featured_knobs)
    X_test_rowlabels = get_exp_labels(X_test, tuner.featured_knobs)
    X_test = Matrix(X_test, X_test_rowlabels, tuner.featured_knobs)
    
    # Scale X_train, y_train and X_test
    X_standardizer = Standardize()
    y_standardizer = Standardize()
    X_train.data = X_standardizer.fit_transform(X_train.data)
    y_train.data = y_standardizer.fit_transform(y_train.data)
    X_test.data = X_standardizer.fit_transform(X_test.data)

    # Make predictions
    ypreds, sigmas, eips = predict(X_train,
                                   y_train,
                                   X_test,
                                   ridge,
                                   tuner.optimization_metric,
                                   tuner.engine)
    

def predict(X_train, y_train, X_test, ridge, metric, eng):
    from common.timeutil import stopwatch
#     #gp_model = self._models[metric]
#     #X_mean,X_std,y_mean,y_std=self.get_scale_params(metric)
#     X_train = self.X()[self._train_indices]
#     n_train_samples,nfeats = X_train.shape
#     #assert n_train_samples <= self._MAX_TRAIN_SAMPLES
#     
#     X_train_scaled,X_mean,X_std = gpr.scale_data(X_train)
#     print "eng: {}".format(eng)
#     print "X_train shape: {}".format(X_train_scaled.shape)
#     X_train_scaled = X_train_scaled.ravel()
#     print "X train shape after ravel: {}".format(X_train_scaled.shape)
#     if X_train_scaled.ndim > 1:
#         X_train_scaled = X_train_scaled.A1
#     print "X train shape after A1: {}".format(X_train_scaled.shape)
#     X_train_scaled = X_train_scaled.tolist()
#     y_train = self.y(metric)[self._train_indices]
#     y_train_scaled,y_mean,y_std = gpr.scale_data(y_train)
#     print "y_train shape: {}".format(y_train_scaled.shape)
#     if y_train_scaled.ndim > 1:
#         y_train_scaled = y_train_scaled.ravel()
#         print "y train shape after ravel: {}".format(y_train_scaled.shape)
#     y_train_scaled = y_train_scaled.tolist()
#     print "X shape: {}, X_mean: {}, X_std: {}".format(X.shape,X_mean,X_std)
#     X_scaled,_,_ = gpr.scale_data(X,X_mean,X_std)
#     n_entries = n_train_samples * nfeats
#     print "X shape: {}".format(X_scaled.shape)
#     X_scaled = X_scaled.ravel()
#     print "X shape after ravel: {}".format(X_scaled.shape)
#     if X_scaled.ndim > 1:
#         X_scaled = X_scaled.A1
#     print "X shape after A1: {}".format(X_scaled.shape)
#     X_scaled = X_scaled.tolist()
#     if self._ridge is None:
#         ridge = 0.001*np.ones((n_train_samples,))
#     else:
#         ridge = self._ridge
#     
#     print "Ridge shape: {}".format(ridge.shape)
#     ridge = ridge.tolist()
    
    print "X_train shape: {}".format(X_train.shape)
    print "y_train shape: {}".format(y_train.shape)
    print "X_test shape: {}".format(X_test.shape)
    print "Ridge shape: {}".format(ridge.shape)
    
    print "\nStarting predictions..."
    with stopwatch("GP predictions"):
        ypreds, sigmas, eips = eng.gp(X_train.tolist(),
                                      y_train.tolist(),
                                      X_test.tolist(),
                                      ridge.tolist(),
                                      X_train.shape[0],
                                      nargout=3)
    ypreds = np.array(list(ypreds))
    sigmas = np.array(list(sigmas))
    eips = np.array(list(eips))
    return ypreds, sigmas, eips
#     end = time.time()
#     print "Done. ({} sec)".format(end - start)
    #print "ypreds initial shape: {}, type: {}".format(len(ypreds),type(ypreds))
    #print "sigmas initial shape: {}, type: {}".format(len(sigmas),type(sigmas))

    #print "ypreds shape: {}".format(ypreds.shape)

    #print "sigmas shape: {}".format(sigmas.shape)
#     print "\nSample preds: {}".format(metric)
#     y_samps = ypreds[:5]*y_std+y_mean
#     for y_samp in y_samps:
#         print "\ty = {}".format(y_samp)

