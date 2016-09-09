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
