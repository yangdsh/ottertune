'''
Created on Aug 18, 2016

@author: Bohan Zhang
'''

import numpy as np
import tensorflow as tf
from sklearn.utils.validation import check_X_y, check_array, assert_all_finite

NUM_CORES = 2

class GPR(object):
    
    def __init__(self, length_scale=1.0, magnitude=1.0, ridge=1.0,
                 batch_size=3000):
        self.length_scale = length_scale
        self.magnitude = magnitude
        self.ridge = ridge
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.xy_ = None
        self.K = None
    
    def __repr__(self):
        rep = ""
        for k, v in sorted(self.__dict__.iteritems()):
            rep += "{} = {}\n".format(k, v)
        return rep
    
    def __str__(self):
        return self.__repr__()
    
    def fit(self, X_train, y_train):
        self.__reset()
        X_train, y_train = check_X_y(X_train, y_train, multi_output=True,
                                     allow_nd=True, y_numeric=True,
                                     estimator="GPR")
        sample_size = X_train.shape[0]
        self.X_train = np.float32(X_train)
        self.y_train = np.float32(y_train)
        if np.isscalar(self.ridge):
            ridge = np.ones(sample_size, dtype=np.float32) * self.ridge
        else:
            assert self.ridge.size == sample_size
            ridge = np.float32(self.ridge)
        v1 = tf.placeholder(tf.float32, name="v1")
        v2 = tf.placeholder(tf.float32, name="v2")
        dist_op = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(v1, v2), 2), 1))
        X_dists = np.zeros([sample_size, sample_size])
        try:
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            for i in range(sample_size):
                X_dists[i] = sess.run(dist_op, feed_dict={v1:self.X_train[i], v2:self.X_train})
        finally:
            sess.close()
        X_dists = tf.cast(X_dists, tf.float32)
        self.K = self.magnitude * tf.exp(-X_dists / self.length_scale) + tf.diag(ridge);
        self.xy_ = tf.matmul(tf.matrix_inverse(self.K), self.y_train)
        self.trained = True
        return self
    
    def predict(self, X_test):
        if not self.__check_trained():
            raise Exception("The model must be trained before making predictions!")
        X_test = check_array(X_test, allow_nd=True, estimator="GPR")
        test_size = X_test.shape[0]
        sample_size = self.X_train.shape[0]
        X_test = np.float32(X_test)

        v1 = tf.placeholder(tf.float32, name="v1")
        v2 = tf.placeholder(tf.float32, name="v2")
        dist_op = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(v1, v2), 2),1))

        K2 = tf.placeholder(tf.float32, name="K2")
        K3 = tf.placeholder(tf.float32, name="K3")

        yhat_ =  tf.cast(tf.matmul( tf.transpose(K2), self.xy_), tf.float32);
        sig_val = tf.cast((tf.sqrt(tf.diag_part(K3 -  tf.matmul(tf.transpose(K2),
                                                                tf.matmul(tf.matrix_inverse(self.K),
                                                                          K2))))),tf.float32)
        u = tf.placeholder(tf.float32, name="u")
        phi1 = 0.5 * tf.erf(u / np.sqrt(2.0)) + 0.5
        phi2 = (1.0 /np.sqrt(2.0 * np.pi)) * tf.exp(tf.square(u) * (-0.5))
        eip =  tf.mul(u , phi1) + phi2

        arr_offset = 0
        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])
        eips = np.zeros([test_size, 1])
        y_best = tf.cast(tf.reduce_min(self.y_train, 0, True), tf.float32)
        try:
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            while arr_offset < test_size:
                if arr_offset + self.batch_size > test_size:
                    end_offset = test_size
                else:
                    end_offset = arr_offset + self.batch_size;
    
                X_test_batch = X_test[arr_offset:end_offset];
                batch_len = end_offset - arr_offset
        
                dists = np.zeros([sample_size,batch_len])
                for i in range(sample_size):
                    dists[i] = sess.run(dist_op, feed_dict={v1:self.X_train[i], v2:X_test_batch})
        
                K2_ = self.magnitude * tf.exp(-dists / self.length_scale);
                K2_ = sess.run(K2_)
        
                dists = np.zeros([batch_len,batch_len])
                for i in range(batch_len):
                    dists[i] = sess.run(dist_op, feed_dict={v1:X_test_batch[i], v2:X_test_batch})
                K3_ = self.magnitude * tf.exp(-dists / self.length_scale);
                K3_ = sess.run(K3_)
    
                yhat = sess.run(yhat_, feed_dict={K2:K2_})
        
                sigma = np.zeros([1,batch_len],np.float32)
                sigma[0] = (sess.run(sig_val,feed_dict={K2:K2_, K3:K3_}))
                sigma = np.transpose(sigma)
        
                u_ = tf.cast(tf.div(tf.sub(y_best, yhat) , sigma), tf.float32)
                u_ = sess.run(u_)
                eip_p = sess.run(eip, feed_dict = {u:u_})
                eip_ = tf.mul(sigma,eip_p) 
                yhats[arr_offset:end_offset] = yhat
                sigmas[arr_offset:end_offset] =  sigma
                eips[arr_offset:end_offset] = sess.run(eip_)
                arr_offset = end_offset
        finally:
            sess.close()
        assert_all_finite(yhats)
        assert_all_finite(sigmas)
        assert_all_finite(eips)
    
        return yhats, sigmas, eips
    
    def get_params(self, deep=True):
        return {"length_scale": self.length_scale,
                "magnitude": self.magnitude,
                "ridge": self.ridge,
                "batch_size": self.batch_size}
    
    def set_params(self, **parameters):
        for param, val in parameters.iteritems():
            setattr(self, param, val)
        return self
    
    def __check_trained(self):
        if self.X_train is None or self.y_train is None \
                or self.xy_ is None or self.K is None:
            return False
        return True

    def __reset(self):
        self.X_train = None
        self.y_train = None
        self.xy_ = None
        self.K = None

def gp_tf(X_train, y_train, X_test, ridge, length_scale, magnitude, batch_size=3000):
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

    return yhats.ravel(), sigmas.ravel(), eips.ravel()

def main():
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(GPR)

if __name__ == "__main__":
    main()
