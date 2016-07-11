'''
Created on Jul 11, 2016

@author: dvanaken
'''

def get_next_config(workload_name):
    pass

def predict(X_train, y_train, X_test, metric, eng):
    #gp_model = self._models[metric]
    #X_mean,X_std,y_mean,y_std=self.get_scale_params(metric)
    X_train = self.X()[self._train_indices]
    n_train_samples,nfeats = X_train.shape
    #assert n_train_samples <= self._MAX_TRAIN_SAMPLES
    
    X_train_scaled,X_mean,X_std = gpr.scale_data(X_train)
    print "eng: {}".format(eng)
    print "X_train shape: {}".format(X_train_scaled.shape)
    X_train_scaled = X_train_scaled.ravel()
    print "X train shape after ravel: {}".format(X_train_scaled.shape)
    if X_train_scaled.ndim > 1:
        X_train_scaled = X_train_scaled.A1
    print "X train shape after A1: {}".format(X_train_scaled.shape)
    X_train_scaled = X_train_scaled.tolist()
    y_train = self.y(metric)[self._train_indices]
    y_train_scaled,y_mean,y_std = gpr.scale_data(y_train)
    print "y_train shape: {}".format(y_train_scaled.shape)
    if y_train_scaled.ndim > 1:
        y_train_scaled = y_train_scaled.ravel()
        print "y train shape after ravel: {}".format(y_train_scaled.shape)
    y_train_scaled = y_train_scaled.tolist()
    print "X shape: {}, X_mean: {}, X_std: {}".format(X.shape,X_mean,X_std)
    X_scaled,_,_ = gpr.scale_data(X,X_mean,X_std)
    n_entries = n_train_samples * nfeats
    print "X shape: {}".format(X_scaled.shape)
    X_scaled = X_scaled.ravel()
    print "X shape after ravel: {}".format(X_scaled.shape)
    if X_scaled.ndim > 1:
        X_scaled = X_scaled.A1
    print "X shape after A1: {}".format(X_scaled.shape)
    X_scaled = X_scaled.tolist()
    if self._ridge is None:
        ridge = 0.001*np.ones((n_train_samples,))
    else:
        ridge = self._ridge
    
    print "Ridge shape: {}".format(ridge.shape)
    ridge = ridge.tolist()
    
    print "X_train shape: {}".format(len(X_train_scaled))
    print "y_train shape: {}".format(len(y_train_scaled))
    print "X shape: {}".format(len(X_scaled))
    print "Ridge shape: {}".format(len(ridge))

    print "\nStarting predictions..."
    start = time.time()
    ypreds,sigmas,eips = eng.gp(X_train_scaled,y_train_scaled,X_scaled,ridge,nfeats,nargout=3)
    end = time.time()
    print "Done. ({} sec)".format(end - start)
    #print "ypreds initial shape: {}, type: {}".format(len(ypreds),type(ypreds))
    #print "sigmas initial shape: {}, type: {}".format(len(sigmas),type(sigmas))
    ypreds = np.array(list(ypreds))
    #print "ypreds shape: {}".format(ypreds.shape)
    sigmas = np.array(list(sigmas))
    eips = np.array(list(eips))
    #print "sigmas shape: {}".format(sigmas.shape)
    print "\nSample preds: {}".format(metric)
    y_samps = ypreds[:5]*y_std+y_mean
    for y_samp in y_samps:
        print "\ty = {}".format(y_samp)
    return ypreds,sigmas,eips,y_mean,y_std