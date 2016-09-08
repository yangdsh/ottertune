'''
Created on Aug 18, 2016

@author: Bohan Zhang, Dana Van Aken
'''
from collections import namedtuple
import numpy as np
import tensorflow as tf
from time import time

# NUM_CORES = 2
# 
# class GPR(object):
#     
#     def __init__(self, length_scale=1.0, magnitude=1.0, ridge=1.0,
#                  batch_size=3000, max_train_size=5000):
#         self.length_scale = length_scale
#         self.magnitude = magnitude
#         self.ridge = ridge
#         self.batch_size = batch_size
#         self.max_train_size = max_train_size
#         self.X_train = None
#         self.y_train = None
#         self.xy_ = None
#         self.K = None
#     
#     def __repr__(self):
#         rep = ""
#         for k, v in sorted(self.__dict__.iteritems()):
#             rep += "{} = {}\n".format(k, v)
#         return rep
#     
#     def __str__(self):
#         return self.__repr__()
#     
#     def fit(self, X_train, y_train):
#         self.__reset()
#         if X_train.shape[0] > self.max_train_size:
#             raise Exception("X_train size cannot exceed {}"
#                             .format(self.max_train_size))
# #         X_train, y_train = check_X_y(X_train, y_train, multi_output=True,
# #                                      allow_nd=True, y_numeric=True,
# #                                      estimator="GPR")
#         sample_size = X_train.shape[0]
#         self.X_train = np.float32(X_train)
#         self.y_train = np.float32(y_train)
#         if np.isscalar(self.ridge):
#             ridge = np.ones(sample_size, dtype=np.float32) * self.ridge
#         else:
#             assert self.ridge.size == sample_size
#             ridge = np.float32(self.ridge)
#         v1 = tf.placeholder(tf.float32, name="v1")
#         v2 = tf.placeholder(tf.float32, name="v2")
#         dist_op = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(v1, v2), 2), 1))
#         X_dists = np.zeros([sample_size, sample_size])
#         try:
#             sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#             for i in range(sample_size):
#                 X_dists[i] = sess.run(dist_op, feed_dict={v1:self.X_train[i], v2:self.X_train})
#         finally:
#             sess.close()
#         X_dists = tf.cast(X_dists, tf.float32)
#         self.K = self.magnitude * tf.exp(-X_dists / self.length_scale) + tf.diag(ridge);
#         self.xy_ = tf.matmul(tf.matrix_inverse(self.K), self.y_train)
#         self.trained = True
#         return self
#     
#     def predict(self, X_test):
#         if not self.__check_trained():
#             raise Exception("The model must be trained before making predictions!")
#         #X_test = check_array(X_test, allow_nd=True, estimator="GPR")
#         test_size = X_test.shape[0]
#         sample_size = self.X_train.shape[0]
#         X_test = np.float32(X_test)
# 
#         v1 = tf.placeholder(tf.float32, name="v1")
#         v2 = tf.placeholder(tf.float32, name="v2")
#         dist_op = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(v1, v2), 2),1))
# 
#         K2 = tf.placeholder(tf.float32, name="K2")
#         K3 = tf.placeholder(tf.float32, name="K3")
# 
#         yhat_ =  tf.cast(tf.matmul( tf.transpose(K2), self.xy_), tf.float32);
#         sig_val = tf.cast((tf.sqrt(tf.diag_part(K3 -  tf.matmul(tf.transpose(K2),
#                                                                 tf.matmul(tf.matrix_inverse(self.K),
#                                                                           K2))))),tf.float32)
#         u = tf.placeholder(tf.float32, name="u")
#         phi1 = 0.5 * tf.erf(u / np.sqrt(2.0)) + 0.5
#         phi2 = (1.0 /np.sqrt(2.0 * np.pi)) * tf.exp(tf.square(u) * (-0.5))
#         eip =  tf.mul(u , phi1) + phi2
# 
#         arr_offset = 0
#         yhats = np.zeros([test_size, 1])
#         sigmas = np.zeros([test_size, 1])
#         eips = np.zeros([test_size, 1])
#         y_best = tf.cast(tf.reduce_min(self.y_train, 0, True), tf.float32)
#         try:
#             sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#             while arr_offset < test_size:
#                 if arr_offset + self.batch_size > test_size:
#                     end_offset = test_size
#                 else:
#                     end_offset = arr_offset + self.batch_size;
#     
#                 X_test_batch = X_test[arr_offset:end_offset];
#                 batch_len = end_offset - arr_offset
#         
#                 dists = np.zeros([sample_size,batch_len])
#                 for i in range(sample_size):
#                     dists[i] = sess.run(dist_op, feed_dict={v1:self.X_train[i], v2:X_test_batch})
#         
#                 K2_ = self.magnitude * tf.exp(-dists / self.length_scale);
#                 K2_ = sess.run(K2_)
#         
#                 dists = np.zeros([batch_len,batch_len])
#                 for i in range(batch_len):
#                     dists[i] = sess.run(dist_op, feed_dict={v1:X_test_batch[i], v2:X_test_batch})
#                 K3_ = self.magnitude * tf.exp(-dists / self.length_scale);
#                 K3_ = sess.run(K3_)
#     
#                 yhat = sess.run(yhat_, feed_dict={K2:K2_})
#         
#                 sigma = np.zeros([1,batch_len],np.float32)
#                 sigma[0] = (sess.run(sig_val,feed_dict={K2:K2_, K3:K3_}))
#                 sigma = np.transpose(sigma)
#         
#                 u_ = tf.cast(tf.div(tf.sub(y_best, yhat) , sigma), tf.float32)
#                 u_ = sess.run(u_)
#                 eip_p = sess.run(eip, feed_dict = {u:u_})
#                 eip_ = tf.mul(sigma,eip_p) 
#                 yhats[arr_offset:end_offset] = yhat
#                 sigmas[arr_offset:end_offset] =  sigma
#                 eips[arr_offset:end_offset] = sess.run(eip_)
#                 arr_offset = end_offset
#         finally:
#             sess.close()
# #         assert_all_finite(yhats)
# #         assert_all_finite(sigmas)
# #         assert_all_finite(eips)
#     
#         return yhats, sigmas, eips
#     
#     def get_params(self, deep=True):
#         return {"length_scale": self.length_scale,
#                 "magnitude": self.magnitude,
#                 "ridge": self.ridge,
#                 "batch_size": self.batch_size}
#     
#     def set_params(self, **parameters):
#         for param, val in parameters.iteritems():
#             setattr(self, param, val)
#         return self
#     
#     def __check_trained(self):
#         if self.X_train is None or self.y_train is None \
#                 or self.xy_ is None or self.K is None:
#             return False
#         return True
# 
#     def __reset(self):
#         self.X_train = None
#         self.y_train = None
#         self.xy_ = None
#         self.K = None

GPRResult = namedtuple('GPRResult', ['ypreds', 'sigmas', 'minL', 'minL_conf'])

class GPR(object):
    
    MAX_TRAIN_SIZE = 5000
    BATCH_SIZE = 3000
    
    def __init__(self, length_scale=1.0, magnitude=1.0):
        self.length_scale = length_scale
        self.magnitude = magnitude
        self.X_train = None
        self.y_train = None
        self.xy_ = None
        self.K = None
        self.graph = None
        self.vars = None
        self.ops = None

    def build_graph(self):
        self.vars = {}
        self.ops = {}
        self.graph = tf.Graph()
        with self.graph.as_default():
            mag_const = tf.constant(self.magnitude,
                                    dtype=np.float32,
                                    name='magnitude')
            ls_const = tf.constant(self.length_scale,
                                   dtype=np.float32,
                                   name='length_scale')

            # Nodes for distance computation
            v1 = tf.placeholder(tf.float32, name="v1")
            v2 = tf.placeholder(tf.float32, name="v2")
            dist_op = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(v1, v2), 2), 1), name='dist_op')
            
            self.vars['v1_h'] = v1
            self.vars['v2_h'] = v2
            self.ops['dist_op'] = dist_op
            
            # Nodes for kernel computation
            X_dists = tf.placeholder(tf.float32, name='X_dists')
            ridge_ph = tf.placeholder(tf.float32, name='ridge')
            K_op = mag_const * tf.exp(-X_dists / ls_const)
            K_ridge_op = K_op + tf.diag(ridge_ph)
            #K_casted_op = tf.cast(K_op, tf.float32)
            
            self.vars['X_dists_h'] = X_dists
            self.vars['ridge_h'] = ridge_ph
            self.ops['K_op'] = K_op
            #self.ops['K_casted_op'] = K_casted_op
            self.ops['K_ridge_op'] = K_ridge_op
            
            # Nodes for xy computation
            K = tf.placeholder(tf.float32, name='K')
            xy_ = tf.placeholder(tf.float32, name='xy_')
            yt_ = tf.placeholder(tf.float32, name='yt_')
            xy_op = tf.matmul(tf.matrix_inverse(K), yt_)
            
            self.vars['K_h'] = K
            self.vars['xy_h'] = xy_
            self.vars['yt_h'] = yt_
            self.ops['xy_op'] = xy_op
    
            # Nodes for yhat/sigma computation
            K2 = tf.placeholder(tf.float32, name="K2")
            K3 = tf.placeholder(tf.float32, name="K3")
            yhat_ =  tf.cast(tf.matmul( tf.transpose(K2), xy_), tf.float32);
            sv1 = tf.matmul(tf.transpose(K2), tf.matmul(tf.matrix_inverse(K), K2))
            sig_val = tf.cast((tf.sqrt(tf.diag_part(K3 - sv1))), tf.float32)
            #sig_val2 = tf.cast((tf.sqrt(mag_const - sv1)), tf.float32)

            self.vars['K2_h'] = K2
            self.vars['K3_h'] = K3
            self.ops['yhat_op'] = yhat_
            self.ops['sig_op'] = sig_val
            
            # Compute y_best (min y)
            y_best_op = tf.cast(tf.reduce_min(yt_, 0, True), tf.float32)
            self.ops['y_best_op'] = y_best_op
            
#             u = tf.placeholder(tf.float32, name="u")
            sigma = tf.placeholder(tf.float32, name='sigma')
            yhat = tf.placeholder(tf.float32, name='yhat')
#             ybest = tf.placeholder(tf.float32, name='ybest')
#             phi1 = 0.5 * tf.erf(u / np.sqrt(2.0)) + 0.5
#             phi2 = (1.0 /np.sqrt(2.0 * np.pi)) * tf.exp(tf.square(u) * (-0.5))
#             eip =  tf.mul(u , phi1) + phi2
#             u_op = tf.cast(tf.div(tf.sub(ybest, yhat), sigma), tf.float32)
            
#             self.vars['u_h'] = u
            self.vars['sigma_h'] = sigma
            self.vars['yhat_h'] = yhat
#             self.vars['ybest_h'] = ybest
#             self.ops['eip_op'] = eip
#             self.ops['u_op'] = u_op

            # Gradiant descent ops/vars

#             K2_mat =  tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(xt_, self.X_train), 2),1))
#             K2_mat = tf.transpose(tf.expand_dims(K2_mat,0))
#             K2_k = tf.cast(self.magnitude * tf.exp(-K2_mat/self.length_scale),tf.float32)
#             #x = tf.matmul(tf.matrix_inverse(self.K) , self.y_train)
#             yhat_gd =  tf.cast(tf.matmul( tf.transpose(K2_k) , self.xy_),tf.float32)
#             sig_val2 = tf.cast((tf.sqrt(self.magnitude -  tf.matmul( tf.transpose(K2_k) ,tf.matmul(tf.matrix_inverse(self.K) , K2_k)) )),tf.float32)
#             

#             self.ops['sig2_op'] = sig_val2
#             self.ops['yhat_gd_op'] = yhat_gd

    def __repr__(self):
        rep = ""
        for k, v in sorted(self.__dict__.iteritems()):
            rep += "{} = {}\n".format(k, v)
        return rep
    
    def __str__(self):
        return self.__repr__()
    
    def check_X_y(self, X, y):
        from sklearn.utils.validation import check_X_y
        
        if X.shape[0] > GPR.MAX_TRAIN_SIZE:
            raise Exception("X_train size cannot exceed {}"
                            .format(self.max_train_size))
        return check_X_y(X, y, multi_output=True,
                         allow_nd=True, y_numeric=True,
                         estimator="GPR")
    
    def check_fitted(self):
        if self.X_train is None or self.y_train is None \
                or self.xy_ is None or self.K is None:
            raise Exception("The model must be trained before making predictions!")
        
    def check_array(self, X):
        from sklearn.utils.validation import check_array
        return check_array(X, allow_nd=True, estimator="GPR")
    
    def check_output(self, X):
        finite_els = np.isfinite(X)
        if not np.all(finite_els):
            raise Exception("Input contains non-finite values: {}"
                            .format(X[~finite_els]))
    
    def fit(self, X_train, y_train, ridge=1.0):
        self.__reset()
        X_train, y_train = self.check_X_y(X_train, y_train)
        self.X_train = np.float32(X_train)
        self.y_train = np.float32(y_train)
        sample_size = self.X_train.shape[0]
        
        if np.isscalar(ridge):
            ridge = np.ones(sample_size) * ridge
        assert ridge.ndim == 1

        X_dists = np.zeros((sample_size, sample_size), dtype=np.float32)
        with tf.Session(graph=self.graph) as sess:
            dist_op = self.ops['dist_op']
            v1, v2 = self.vars['v1_h'], self.vars['v2_h']
            for i in range(sample_size):
                X_dists[i] = sess.run(dist_op, feed_dict={v1:self.X_train[i], v2:self.X_train})
        
            K_ridge_op = self.ops['K_ridge_op']
            X_dists_ph = self.vars['X_dists_h']
            ridge_ph = self.vars['ridge_h']

            self.K = sess.run(K_ridge_op, feed_dict={X_dists_ph:X_dists, ridge_ph:ridge})
            
            xy_op = self.ops['xy_op']
            K_ph = self.vars['K_h']
            yt_ph = self.vars['yt_h']
            self.xy_ = sess.run(xy_op, feed_dict={K_ph:self.K, yt_ph:self.y_train})
            
            # Setup for gradient descent
            print "Setting up for gradient descent"
            start = time()
            xt_ = tf.Variable(self.X_train[0], tf.float32)
            init = tf.initialize_all_variables()
            sess.run(init)
            K2_mat =  tf.transpose(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(xt_, self.X_train), 2),1)), 0))
            K2__ = tf.cast(self.magnitude * tf.exp(-K2_mat/self.length_scale),tf.float32)
            yhat_gd =  tf.cast(tf.matmul( tf.transpose(K2__) , self.xy_),tf.float32)
            sig_val2 = tf.cast((tf.sqrt(self.magnitude -  tf.matmul( tf.transpose(K2__) ,tf.matmul(tf.matrix_inverse(self.K) , K2__)) )),tf.float32)
            self.ops['yhat_gd'] = yhat_gd
            self.ops['sig_val2'] = sig_val2
            self.vars['xt_'] = xt_
            print "Done. {0:.3f} seconds\n".format(time() - start)
        return self
    
    def predict(self, X_test, run_gradient_descent=False):
        self.check_fitted()
        X_test = np.float32(self.check_array(X_test))
        test_size = X_test.shape[0]
        sample_size, nfeats = self.X_train.shape

        arr_offset = 0
        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])
        if run_gradient_descent:
            minLs = np.zeros([test_size, 1])
            minL_confs = np.zeros([test_size, nfeats])
        else:
            minLs = None
            minL_confs = None
        with tf.Session(graph=self.graph) as sess:
            # Nodes for distance operation
            dist_op = self.ops['dist_op']
            v1 = self.vars['v1_h']
            v2 = self.vars['v2_h']
            
            # Nodes for kernel computation
            K_op = self.ops['K_op']
            X_dists = self.vars['X_dists_h']
            
            # Nodes to compute yhats/sigmas
            yhat_ = self.ops['yhat_op']
            K_ph = self.vars['K_h']
            K2 = self.vars['K2_h']
            K3 = self.vars['K3_h']
            xy_ph = self.vars['xy_h']


            while arr_offset < test_size:
                if arr_offset + GPR.BATCH_SIZE > test_size:
                    end_offset = test_size
                else:
                    end_offset = arr_offset + GPR.BATCH_SIZE;
    
                X_test_batch = X_test[arr_offset:end_offset];
                batch_len = end_offset - arr_offset
        
                dists1 = np.zeros([sample_size,batch_len])
                for i in range(sample_size):
                    dists1[i] = sess.run(dist_op, feed_dict={v1:self.X_train[i], v2:X_test_batch})

                if run_gradient_descent:
                    max_iter = 0
                    learning_rate = 0.1
                    xt_ = self.vars['xt_']
                    print "Initializing variables"
                    start = time()
                    init = tf.initialize_all_variables()
                    sess.run(init)
                    print "Done. {0:.3f} seconds\n".format(time() - start)
                    
                    print "Initializing optimizer"
                    start = time()
                    sig_val = self.ops['sig_val2']
                    yhat_gd = self.ops['yhat_gd']
                    Loss = tf.squeeze(tf.sub(yhat_gd, sig_val)) 
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                    train = optimizer.minimize(Loss)
                    print "Done. {0:.3f} seconds\n".format(time() - start)

                    yhat = np.empty((batch_len, 1))
                    sigma = np.empty((batch_len, 1))
                    minL = np.empty((batch_len, 1))
                    minL_conf = np.empty((batch_len, nfeats))
                    for i in range(batch_len):
                        print "Predicting test input {}/{}".format(i+1,test_size)
                        assign_op = xt_.assign(X_test_batch[i])
                        sess.run(assign_op) 
                        for step in range(max_iter):
                            print i, step,  sess.run(Loss)
                            sess.run(train)
                        print "computing yhat"
                        yhat[i] = sess.run(yhat_gd)[0][0]
                        print "done"
                        print "computing sigma"
                        sigma[i] = sess.run(sig_val)[0][0]
                        print "done"
                        print "computing minL"
                        minL[i] = sess.run(Loss)
                        print "done"
                        print "computing minL_conf"
                        minL_conf[i] = sess.run(xt_)
                        print "done"
                    minLs[arr_offset:end_offset] = minL
                    minL_confs[arr_offset:end_offset] = minL_conf
                else:
                    sig_val = self.ops['sig_op']
                    K2_ = sess.run(K_op, feed_dict={X_dists:dists1})
                    yhat = sess.run(yhat_, feed_dict={K2:K2_, xy_ph:self.xy_})
                    dists2 = np.zeros([batch_len,batch_len])
                    for i in range(batch_len):
                        dists2[i] = sess.run(dist_op, feed_dict={v1:X_test_batch[i], v2:X_test_batch})
                    K3_ = sess.run(K_op, feed_dict={X_dists:dists2})
            
                    sigma = np.zeros([1,batch_len], np.float32)
                    sigma[0] = sess.run(sig_val,feed_dict={K_ph:self.K, K2:K2_, K3:K3_})
                    sigma = np.transpose(sigma)
                yhats[arr_offset:end_offset] = yhat
                sigmas[arr_offset:end_offset] =  sigma
                arr_offset = end_offset

        self.check_output(yhats)
        self.check_output(sigmas)
        if run_gradient_descent:
            self.check_output(minLs)
            self.check_output(minL_confs)

        return GPRResult(yhats, sigmas, minLs, minL_confs)
    
    def get_params(self, deep=True):
        return {"length_scale": self.length_scale,
                "magnitude": self.magnitude}
    
    def set_params(self, **parameters):
        for param, val in parameters.iteritems():
            setattr(self, param, val)
        return self


    def __reset(self):
        import gc

        self.X_train = None
        self.y_train = None
        self.xy_ = None
        self.K = None
        self.graph = None
        self.build_graph()
        gc.collect()

def gp_tf(X_train, y_train, X_test, ridge, length_scale, magnitude, batch_size=3000):
    with tf.Graph().as_default():
        y_best = tf.cast(tf.reduce_min(y_train, 0, True), tf.float32)
        sample_size = X_train.shape[0]
        train_size = X_test.shape[0]
        arr_offset = 0
        yhats = np.zeros([train_size, 1])
        sigmas = np.zeros([train_size, 1])
        eips = np.zeros([train_size, 1])
        X_train = np.float32(X_train)
        y_train = np.float32(y_train)
        X_test = np.float32(X_test)
        ridge = np.float32(ridge)
    
        v1 = tf.placeholder(tf.float32,name="v1")
        v2 = tf.placeholder(tf.float32,name="v2")
        dist_op = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(v1, v2), 2), 1))
        try:
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        
            dists = np.zeros([sample_size,sample_size])
            for i in range(sample_size):
                dists[i] = sess.run(dist_op,feed_dict={v1:X_train[i], v2:X_train})
        
        
            dists = tf.cast(dists, tf.float32)
            K = magnitude * tf.exp(-dists/length_scale) + tf.diag(ridge);
        
            K2 = tf.placeholder(tf.float32, name="K2")
            K3 = tf.placeholder(tf.float32, name="K3")
        
            x = tf.matmul(tf.matrix_inverse(K), y_train)
            yhat_ =  tf.cast(tf.matmul(tf.transpose(K2), x), tf.float32);
            sig_val = tf.cast((tf.sqrt(tf.diag_part(K3 -  tf.matmul(tf.transpose(K2),
                                                                    tf.matmul(tf.matrix_inverse(K),
                                                                              K2))))),
                              tf.float32)
    
            u = tf.placeholder(tf.float32, name="u")
            phi1 = 0.5 * tf.erf(u / np.sqrt(2.0)) + 0.5
            phi2 = (1.0 / np.sqrt(2.0 * np.pi)) * tf.exp(tf.square(u) * (-0.5));
            eip = (tf.mul(u, phi1) + phi2);
        
            while arr_offset < train_size:
                if arr_offset + batch_size > train_size:
                    end_offset = train_size
                else:
                    end_offset = arr_offset + batch_size;
        
                xt_ = X_test[arr_offset:end_offset];
                batch_len = end_offset - arr_offset
        
                dists = np.zeros([sample_size, batch_len])
                for i in range(sample_size):
                    dists[i] = sess.run(dist_op, feed_dict={v1:X_train[i], v2:xt_})
        
                K2_ = magnitude * tf.exp(-dists / length_scale);
                K2_ = sess.run(K2_)
        
                dists = np.zeros([batch_len, batch_len])
                for i in range(batch_len):
                    dists[i] = sess.run(dist_op, feed_dict={v1:xt_[i], v2:xt_})
                K3_ = magnitude * tf.exp(-dists / length_scale);
                K3_ = sess.run(K3_)
        
                yhat = sess.run(yhat_, feed_dict={K2:K2_})
        
                sigma = np.zeros([1, batch_len], np.float32)
                sigma[0] = (sess.run(sig_val, feed_dict={K2:K2_, K3:K3_}))
                sigma = np.transpose(sigma)
        
                u_ = tf.cast(tf.div(tf.sub(y_best, yhat), sigma), tf.float32)
                u_ = sess.run(u_)
                eip_p = sess.run(eip, feed_dict={u:u_})
                eip_ = tf.mul(sigma, eip_p) 
                yhats[arr_offset:end_offset] = yhat
                sigmas[arr_offset:end_offset] =  sigma;
                eips[arr_offset:end_offset] = sess.run(eip_);
                arr_offset = end_offset
            
        finally:
            sess.close()
    
        return yhats, sigmas, eips

def gd_tf(xs, ys, xt, ridge, length_scale, magnitude, max_iter):
    with tf.Graph().as_default():
        y_best = tf.cast(tf.reduce_min(ys,0,True),tf.float32);   #array
        sample_size = xs.shape[0]
        nfeats = xs.shape[1]
        test_size = xt.shape[0]
        arr_offset = 0
        ini_size = xt.shape[0]
    
        yhats = np.zeros([test_size,1])
        sigmas = np.zeros([test_size,1])
        minL = np.zeros([test_size,1])
        new_conf = np.zeros([test_size, nfeats])
        #eips = np.zeros([test_size,1]);
        xs = np.float32(xs)
        ys = np.float32(ys)
        ############## 
        xt_ = tf.Variable(xt[0],tf.float32) 
    
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
    
        ridge = np.float32(ridge)

        v1 = tf.placeholder(tf.float32,name="v1")
        v2 = tf.placeholder(tf.float32,name="v2")
        dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(v1, v2), 2),1))
    
        tmp = np.zeros([sample_size,sample_size])
        for i in range(sample_size):
            tmp[i] = sess.run(dist,feed_dict={v1:xs[i],v2:xs})
        print "Finished euc matrix \n"
    
    
        tmp = tf.cast(tmp,tf.float32)
        K = magnitude * tf.exp(-tmp/length_scale) + tf.diag(ridge);
        print "Finished K "
    
        K2_mat =  tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(xt_, xs), 2),1))
        K2_mat = tf.transpose(tf.expand_dims(K2_mat,0))
        K2 = tf.cast(magnitude * tf.exp(-K2_mat/length_scale),tf.float32)
    
        x = tf.matmul(tf.matrix_inverse(K) , ys)
        yhat_ =  tf.cast(tf.matmul( tf.transpose(K2) ,x),tf.float32)
        sig_val = tf.cast((tf.sqrt(magnitude -  tf.matmul( tf.transpose(K2) ,tf.matmul(tf.matrix_inverse(K) , K2)) )),tf.float32)
    
        Loss = tf.squeeze(tf.sub(yhat_,sig_val))
    #    optimizer = tf.train.GradientDescentOptimizer(0.1)    
        optimizer = tf.train.AdamOptimizer(0.1)
        train = optimizer.minimize(Loss)
        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(ini_size):
            assign_op = xt_.assign(xt[i])
            sess.run(assign_op) 
            for step in range(max_iter):
                print i, step,  sess.run(Loss)
                sess.run(train)
            yhats[i] = sess.run(yhat_)[0][0]
            sigmas[i] = sess.run(sig_val)[0][0]
            minL[i] = sess.run(Loss)
            new_conf[i] = sess.run(xt_)
        return yhats, sigmas, minL, new_conf

def main():
    check_gd_equivalence()
#     X_train, y_train, X_test, length_scale, magnitude, ridge = create_random_matrices(n_test=1000)
#     gpr = GPR2(length_scale, magnitude, ridge)
#     gpr.fit(X_train, y_train)
#     gpr.predict(X_test)

def create_random_matrices(n_samples=3000, n_feats=12, n_test=4444):
    X_train = np.random.rand(n_samples, n_feats)
    y_train = np.random.rand(n_samples, 1)
    X_test = np.random.rand(n_test, n_feats)
    
    length_scale = np.random.rand()
    magnitude = np.random.rand()
    ridge = np.ones(n_samples) * np.random.rand()
    
    return X_train, y_train, X_test, length_scale, magnitude, ridge

def check_equivalence():
    from time import time

    X_train, y_train, X_test, length_scale, magnitude, ridge = create_random_matrices()
    
    print "Running GPR method..."
    start = time()
    yhats1, sigmas1, eips1 = gp_tf(X_train, y_train, X_test, ridge,
                                   length_scale, magnitude)
    print "GPR method: {0:.3f} seconds".format(time() - start)
    
    print "Running GPR class..."
    start = time()
    gpr = GPR(length_scale, magnitude)
    gpr.fit(X_train, y_train, ridge)
    yhats2, sigmas2, eips2 = gpr.predict(X_test)
    print "GPR class: {0:.3f} seconds".format(time() - start)
 
    assert np.allclose(yhats1, yhats2)
    assert np.allclose(sigmas1, sigmas2)
    assert np.allclose(eips1, eips2)

def check_gd_equivalence():
    X_train, y_train, X_test, length_scale, magnitude, ridge = create_random_matrices(n_test=5)

#     print "Running GPR method..."
#     start = time()
#     yhats3, sigmas3, _ = gp_tf(X_train, y_train, X_test, ridge,
#                                length_scale, magnitude)
#     print "Done."
#     print "GPR method: {0:.3f} seconds\n".format(time() - start)
      
    print "Running GD method..."
    start = time()
    yhats1, sigmas1, minL, minL_conf = gd_tf(X_train, y_train, X_test, ridge,
                                  length_scale, magnitude, max_iter=0)
    print "Done."
    print "GD method: {0:.3f} seconds\n".format(time() - start)
    
    print "Running GPR class..."
    start = time()
    gpr = GPR(length_scale, magnitude)
    gpr.fit(X_train, y_train, ridge)
    gpres = gpr.predict(X_test, run_gradient_descent=True)
    print "GPR class: {0:.3f} seconds\n".format(time() - start)
     
#     print yhats1
#     print gpres.ypreds
# #     print yhats3
#     print ""
#     print sigmas1
#     print gpres.sigmas
# #     print sigmas3
#     print ""
#     print minL
#     print gpres.minL
#     print ""
#     print minL_conf
#     print gpres.minL_conf
#     print ""
#     assert np.allclose(yhats1, yhats3, atol=1e-4)
#     assert np.allclose(sigmas1, sigmas3, atol=1e-4)
    assert np.allclose(yhats1, gpres.ypreds, atol=1e-4)
    assert np.allclose(sigmas1, gpres.sigmas, atol=1e-4)
    assert np.allclose(minL, gpres.minL, atol=1e-4)
    assert np.allclose(minL_conf, gpres.minL_conf, atol=1e-4)
    

 


if __name__ == "__main__":
    main()
