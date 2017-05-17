import tensorflow as tf
import numpy as np
import scipy.io
import sys
sys.path.append('../')
from src.convolutional import set_convolutional

###################################################################
# this is defined manually and should reflect the network to import
conv_stride = np.array([2,1,1,1,1])
filtergroup_yn = np.array([0,1,0,1,1], dtype=bool)
bnorm_yn = np.array([1,1,1,1,0], dtype=bool)
relu_yn = np.array([1,1,1,1,0], dtype=bool)
pool_stride = np.array([2,1,0,0,0]) # 0 means no pool
pool_sz = 3
bnorm_adjust = True
assert len(conv_stride) == len(filtergroup_yn) == len(bnorm_yn) == len(relu_yn) == len(pool_stride), ('These arrays of flags should have same length')
assert all(conv_stride) >= True, ('The number of conv layers is assumed to define the depth of the network')
num_layers = len(conv_stride)
###################################################################

# import pretrained Siamese network from matconvnet
def siamese(net_path, X, Z):
    # read mat file from net_path and start TF Siamese graph from placeholders X and Z
    mat = scipy.io.loadmat(net_path)
    net_dot_mat = mat.get('net')

    ## organize parameters to import
    params = net_dot_mat['params']
    params = params[0][0]
    params_names = params['name'][0]
    params_names_list = [params_names[p][0] for p in xrange(params_names.size)]
    params_values = params['value'][0]
    params_values_list = [params_values[p] for p in xrange(params_values.size)]
    template_z, template_x = create_siamese(X, Z, params_names_list, params_values_list)
    return template_z, template_x, params_names_list, params_values_list

# find all parameters matching the codename (there should be only one)
def find_params(x, params):
    matching = [s for s in params if x in s]
    assert len(matching)==1, ('Ambiguous param name found')    
    return matching

# tensorflow graph
def create_siamese(X, Z, params_names_list, params_values_list):
    # placeholders for search region crop X and exemplar crop Z    
    net_x = X    
    net_z = Z
    # loop through the flag arrays and re-construct network, reading parameters of conv and bnorm layers
    for i in xrange(num_layers):
        print '> Layer '+str(i+1)
        # conv
        conv_W_name = find_params('conv'+str(i+1)+'f', params_names_list)[0]
        conv_b_name = find_params('conv'+str(i+1)+'b', params_names_list)[0]
        print '\t\tCONV: setting '+conv_W_name+' '+conv_b_name
        print '\t\tCONV: stride '+str(conv_stride[i])+', filter-group '+str(filtergroup_yn[i])
        conv_W = params_values_list[params_names_list.index(conv_W_name)]
        conv_b = params_values_list[params_names_list.index(conv_b_name)]
        # batchnorm
        if bnorm_yn[i]:
            bn_beta_name = find_params('bn'+str(i+1)+'b', params_names_list)[0]
            bn_gamma_name = find_params('bn'+str(i+1)+'m', params_names_list)[0]
            bn_moments_name = find_params('bn'+str(i+1)+'x', params_names_list)[0]
            print '\t\tBNORM: setting '+bn_beta_name+' '+bn_gamma_name+' '+bn_moments_name
            bn_beta = params_values_list[params_names_list.index(bn_beta_name)]
            bn_gamma = params_values_list[params_names_list.index(bn_gamma_name)]
            bn_moments = params_values_list[params_names_list.index(bn_moments_name)]
            bn_moving_mean = bn_moments[:,0]
            bn_moving_variance = bn_moments[:,1]**2 # saved as std in matconvnet
        else:
            bn_beta = bn_gamma = bn_moving_mean = bn_moving_variance = []
        
        # set up conv "block" with bnorm and activation 
        net_x = set_convolutional(net_x, conv_W, np.swapaxes(conv_b,0,1), conv_stride[i], \
                            bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance, \
                            filtergroup=filtergroup_yn[i], batchnorm=bnorm_yn[i], activation=relu_yn[i], \
                            scope='conv'+str(i+1), reuse=False)
        
        # notice reuse=True for Siamese parameters sharing
        net_z = set_convolutional(net_z, conv_W, np.swapaxes(conv_b,0,1), conv_stride[i], \
                            bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance, \
                            filtergroup=filtergroup_yn[i], batchnorm=bnorm_yn[i], activation=relu_yn[i], \
                            scope='conv'+str(i+1), reuse=True)    
        
        # add max pool if required
        if pool_stride[i]>0:
            print '\t\tMAX-POOL: size '+str(pool_sz)+ ' and stride '+str(pool_stride[i])
            net_x = tf.nn.max_pool(net_x, [1,pool_sz,pool_sz,1], strides=[1,pool_stride[i],pool_stride[i],1], padding='VALID', name='pool'+str(i+1))
            net_z = tf.nn.max_pool(net_z, [1,pool_sz,pool_sz,1], strides=[1,pool_stride[i],pool_stride[i],1], padding='VALID', name='pool'+str(i+1))
    
    return net_z, net_x

def match_templates(net_z, net_x, params_names_list, params_values_list):
    ## finalize network
    # z, x are [B, H, W, C]
    net_z = tf.transpose(net_z, perm=[1,2,0,3])
    net_x = tf.transpose(net_x, perm=[1,2,0,3])
    # z, x are [H, W, B, C]
    shape_z = tf.shape(net_z)
    shape_x = tf.shape(net_x)
    Hz = shape_z[0]
    Wz = shape_z[1]
    B = shape_z[2]
    C = shape_z[3]
    Hx = shape_x[0]
    Wx = shape_x[1]
    Bx = shape_x[2]
    Cx = shape_x[3]
    # assert B==Bx, ('Z and X should have same Batch size')
    # assert C==Cx, ('Z and X should have same Channels number')
    net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))
    net_x = tf.reshape(net_x, (1, Hx, Wx, B*C))
    net_final = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID')
    # final is [1, Hf, Wf, BC]
    net_final = tf.concat(tf.split(net_final, 3, axis=3), axis=0)
    # final is [B, Hf, Wf, C]
    net_final = tf.expand_dims(tf.reduce_sum(net_final, axis=3), axis=3)
    # final is [B, Hf, Wf, 1]
    if bnorm_adjust:
        bn_beta = params_values_list[params_names_list.index('fin_adjust_bnb')]
        bn_gamma = params_values_list[params_names_list.index('fin_adjust_bnm')]
        bn_moments = params_values_list[params_names_list.index('fin_adjust_bnx')]
        bn_moving_mean = bn_moments[:,0]
        bn_moving_variance = bn_moments[:,1]**2
        net_final = tf.layers.batch_normalization(net_final, beta_initializer=tf.constant_initializer(bn_beta), \
                                                gamma_initializer=tf.constant_initializer(bn_gamma), \
                                                moving_mean_initializer=tf.constant_initializer(bn_moving_mean), \
                                                moving_variance_initializer=tf.constant_initializer(bn_moving_variance), \
                                                training=False, trainable=False)

    return net_final