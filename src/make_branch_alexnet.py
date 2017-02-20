def make_branch_alexnet():
	X = tf.placeholder(tf.float32)
	X1 = tf.expand_dims(X, 0)
	h1 = convolutional(X1, 11, 11, 3, 96, stride=2, scope='conv1')