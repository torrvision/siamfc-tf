
import tensorflow as tf

def set_convolutional(X, W, b, bn_beta, bn_gamma, bn_mm, bn_mv, stride, batchnorm=True, activation=True, scope=None):
    # use the input scope or default to "conv"
    with tf.variable_scope(scope or 'conv'):
        W = tf.get_variable("W", W.shape, trainable=False, initializer=tf.constant_initializer(W))
        b = tf.get_variable("b", b.shape, trainable=False, initializer=tf.constant_initializer(b))

        h = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='VALID') + b        

        if batchnorm:
            # h = bn_beta + h*bn_gamma
            h = tf.layers.batch_normalization(h,epsilon=1e-5, \
                                                momentum=0.9, \
                                                beta_initializer=tf.constant_initializer(bn_beta), \
                                                gamma_initializer=tf.constant_initializer(bn_gamma), \
                                                moving_mean_initializer=tf.constant_initializer(bn_mm), \
                                                moving_variance_initializer=tf.constant_initializer(bn_mv), \
                                                training=False, trainable=False, reuse=False)

        if activation:
            h = tf.nn.relu(h)

        return h