
def convolutional(X, height, width, n_input, n_output, stride, batchnorm=True, activation=True, scope=None):
    # use the input scope or default to "conv"
    with tf.variable_scope(scope or 'conv'):
        W = tf.get_variable(
            name='W',
            shape=[height, width, n_input, n_output],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer())
        h = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='VALID') + b        
        if batchnorm:
            h = tf.contrib.layers.batch_norm(h, activation_fn=tf.nn.relu, is_training=False, reuse=False)
        else:        
            if activation:
                h = tf.nn.relu(h)
        return h

def set_convolutional(W, b, X, height, width, n_input, n_output, stride, batchnorm=True, activation=True, scope=None):
    # use the input scope or default to "conv"
    with tf.variable_scope(scope or 'conv'):
        W = tf.get_variable(
            name='W',
            shape=[height, width, n_input, n_output],
            initializer=W)
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=b)
        h = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='VALID') + b 
        if batchnorm:
            #TODO
            h = tf.contrib.layers.batch_norm(h, activation_fn=tf.nn.relu, is_training=False, reuse=False)
        else:        
            if activation:
                h = tf.nn.relu(h)
        return h        