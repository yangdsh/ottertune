import numpy as np
import tensorflow as tf

def gp_tf(xs,ys,xt,ridge):

    MAX_ITER = 20
    batch_size = 3000
    sigma_cl = 1.07;
    y_best = tf.cast(tf.reduce_min(ys,0,True),tf.float32);   #array
    sample_size = xs.shape[0]
    nfeats = xs.shape[1]
    train_size = 1
    arr_offset = 0
    ini_size = xt.shape[0]

    yhats = np.zeros([train_size,1]);
    sigmas = np.zeros([train_size,1]);
    eips = np.zeros([train_size,1]);
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
    K = tf.exp(-tmp/sigma_cl) + tf.diag(ridge);
    print "Finished K "

    K2_mat =  tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(xt_, xs), 2),1))
    K2_mat = tf.transpose(tf.expand_dims(K2_mat,0))
    K2 = tf.cast(tf.exp(-K2_mat/sigma_cl),tf.float32)

    x = tf.matmul(tf.matrix_inverse(K) , ys)
    yhat_ =  tf.cast(tf.matmul( tf.transpose(K2) ,x),tf.float32)
    sig_val = tf.cast((tf.sqrt(1 -  tf.matmul( tf.transpose(K2) ,tf.matmul(tf.matrix_inverse(K) , K2)) )),tf.float32)

    Loss = tf.squeeze(tf.sub(yhat_,sig_val))
#    optimizer = tf.train.GradientDescentOptimizer(0.1)    
    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(Loss)
    init = tf.initialize_all_variables()
    sess.run(init)

    yhats = []
    sigmas = []
    minL = []
    new_conf = []
    for i in range(ini_size):
        assign_op = xt_.assign(xt[i])
        sess.run(assign_op) 
        for step in range(MAX_ITER):
            print i, step,  sess.run(Loss)
            sess.run(train)
        yhats.append(sess.run(yhat_)[0][0])
        sigmas.append(sess.run(sig_val)[0][0])
        minL.append(sess.run(Loss))
        new_conf.append(sess.run(xt_))
    return yhats, sigmas, minL,new_conf

def create_random_matrices(n_samples=3000, n_feats=12, n_test=5):
    X_train = np.random.rand(n_samples, n_feats)
    y_train = np.random.rand(n_samples, 1)
    X_test = np.random.rand(n_test, n_feats)
    
    length_scale = np.random.rand()
    magnitude = np.random.rand()
    ridge = np.ones(n_samples) * np.random.rand()
    
    return X_train, y_train, X_test, length_scale, magnitude, ridge


def test():
    X_train, y_train, X_test, length_scale, magnitude, ridge = create_random_matrices()
    #data = np.load('data.npz')
    #x_train = data['X_train']
    #y_train = data['y_train']
    #x_test =  data['X_test']
    #ridge = data['ridge']
    a,b,c,d = gp_tf(X_train,y_train,X_test,ridge)
    return a,b,c,d 

