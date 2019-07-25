import os,sys
import tensorflow as tf


# Using classic tensorflow.nn.batch_normalization

def IN(X, batch_size, is_train, name=None):

    with tf.variable_scope(name):

        shape = X.get_shape()
        b,h,w,c = X.get_shape().as_list()
        beta, gamma = None, None
        beta = tf.get_variable(name + "Beta", [batch_size,h,w,c], tf.float32, initializer=tf.zeros_initializer(), trainable=True)
        gamma = tf.get_variable(name+ "Gamma", [batch_size,h,w,c], tf.float32, initializer=tf.ones_initializer(), trainable=False)

        mean, variance = tf.nn.moments(X, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        output = tf.nn.batch_normalization(X, mean, variance, beta, gamma, epsilon)
        output.set_shape(shape)
    
        return output


def BN(X, batch_size, is_train, name=None):

    with tf.variable_scope(name):

        shape = X.get_shape()
        b,h,w,c = X.get_shape().as_list()
        beta, gamma = None, None
        beta = tf.get_variable(name + "Beta", [batch_size,h,w,c], tf.float32, initializer=tf.zeros_initializer(), trainable=True)
        gamma = tf.get_variable(name+ "Gamma", [batch_size,h,w,c], tf.float32, initializer=tf.ones_initializer(), trainable=True)

        mean, variance = tf.nn.moments(X, axes=[0,1,2], keep_dims=True)
        epsilon = 1e-5
        output = tf.nn.batch_normalization(X, mean, variance, beta, gamma, epsilon)
        output.set_shape(shape)
    
        return output


#using just tensorflow.nn.batch_normalization
#Whne Using this, You must be added graph.ops update code in model code
#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#with tf.control_dependencies(update_ops):
#    optimizer


def batch_norm(X , is_train = True):

    return tf.layers.batch_normalization(X,momentum=0.9,epsilon=1e-5,training=is_train, trainable=True)

def Inst_norm(X , is_train = True):
    
    return tf.layers.batch_normalization(X,momentum=0.9,axis=[1,2],epsilon=1e-5,training=is_train, trainable=True)



# Using tensorflow.contrib.layers.batch_normalization


def InstNorm(inputT, is_train=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_train,  
                lambda:  tf.contrib.layers.instance_norm(inputT, trainable=True, epsilon= 1e-5, 
                                   center=True, scale=True, scope=scope),  
                lambda:  tf.contrib.layers.instance_norm(inputT, trainable=False, epsilon= 1e-5, 
                                   center=True, scale=True,scope=scope, reuse = True))
    
def BatchNorm(inputT, is_train=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_train,  
                lambda:  tf.contrib.layers.batch_norm(inputT,decay=0.9, is_training=True,  
                                   epsilon= 1e-5,center=True,scale = True, updates_collections=None, scope=scope),  
                lambda:  tf.contrib.layers.batch_norm(inputT,decay=0.9, is_training=False,  
                                   epsilon= 1e-5,updates_collections=None, center=True,scale=True, scope=scope, reuse = True))



# Using very amazing ways to make variables



def _instance_norm(net, train=True):
    batch, rows, cols, channels = net.get_shape().as_list()
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift


def _batch_norm(net, train=True):
    batch, rows, cols, channels = net.get_shape().as_list()
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [0,1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift



