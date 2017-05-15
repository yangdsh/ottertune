import numpy as np
import tensorflow as tf

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
    
    
        tmp = tf.cast(tmp,tf.float32)
        K = magnitude * tf.exp(-tmp/length_scale) + tf.diag(ridge);
    
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
