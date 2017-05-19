import tensorflow as tf
import numpy as np
import scipy.io
import os.path
import sys
sys.path.append('../')
from src.crops import extract_crops_z, extract_crops_x, pad_frame
from src.convolutional import set_convolutional

# Placeholders
pos_x_ph = tf.placeholder(tf.float64)
pos_y_ph = tf.placeholder(tf.float64)
z_sz_ph = tf.placeholder(tf.float64)
x_sz0_ph = tf.placeholder(tf.float64)
x_sz1_ph = tf.placeholder(tf.float64)
x_sz2_ph = tf.placeholder(tf.float64)

# network design, has to reflect the network to import
_conv_stride = np.array([2,1,1,1,1])
_filtergroup_yn = np.array([0,1,0,1,1], dtype=bool)
_bnorm_yn = np.array([1,1,1,1,0], dtype=bool)
_relu_yn = np.array([1,1,1,1,0], dtype=bool)
_pool_stride = np.array([2,1,0,0,0]) # 0 means no pool
_pool_sz = 3
_bnorm_adjust = True
assert len(_conv_stride) == len(_filtergroup_yn) == len(_bnorm_yn) == len(_relu_yn) == len(_pool_stride), ('These arrays of flags should have same length')
assert all(_conv_stride) >= True, ('The number of conv layers is assumed to define the depth of the network')
_num_layers = len(_conv_stride)

# build the TF graph, from list of frames to score maps
def build_tracking_graph(frame_name_list, num_frames, frame_sz, final_score_sz, design, env):
    # Make a queue of file names
    filename_queue = tf.train.string_input_producer(frame_name_list, shuffle=False, capacity=num_frames)
    image_reader = tf.WholeFileReader()
    # Read a whole file from the queue
    _, image_file = image_reader.read(filename_queue)
    # Decode the image as a JPEG file, this will turn it into a Tensor
    image = tf.cast(tf.image.decode_jpeg(image_file), tf.int32)
    # used to pad the crops
    avg_chan = tf.cast(tf.reduce_mean(image, axis=(0,1)), tf.int32)
    # pad with avg color if necessary
    frame_padded_z, npad_z = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, z_sz_ph, avg_chan);
    # extract tensor of z_crops (all identical)
    z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x_ph, pos_y_ph, z_sz_ph, design.exemplar_sz)
    frame_padded_x, npad_x = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, x_sz2_ph, avg_chan);
    # extract tensor of x_crops (3 scales)
    x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x_ph, pos_y_ph, x_sz0_ph, x_sz1_ph, x_sz2_ph, design.search_sz)
    # use crops as input of (MatConvnet imported) pre-trained fully-convolutional Siamese net
    template_z, template_x, p_names_list, p_val_list = branches(os.path.join(env.root_pretrained,design.net), x_crops, z_crops)
    # cross-correlate the two templates and obtain a batch of scores (one per scale)
    scores = match_templates(template_z, template_x, p_names_list, p_val_list)    
    # upsample the score maps
    scores_up = tf.image.resize_images(scores, [final_score_sz, final_score_sz])
    return image, template_z, scores_up

# import pretrained Siamese network from matconvnet
def branches(net_path, X, Z):
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
    template_z, template_x = _create_siamese(X, Z, params_names_list, params_values_list)
    return template_z, template_x, params_names_list, params_values_list

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
    if _bnorm_adjust:
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

# find all parameters matching the codename (there should be only one)
def _find_params(x, params):
    matching = [s for s in params if x in s]
    assert len(matching)==1, ('Ambiguous param name found')    
    return matching

# tensorflow graph
def _create_siamese(X, Z, params_names_list, params_values_list):
    # placeholders for search region crop X and exemplar crop Z    
    net_x = X    
    net_z = Z
    # loop through the flag arrays and re-construct network, reading parameters of conv and bnorm layers
    for i in xrange(_num_layers):
        print '> Layer '+str(i+1)
        # conv
        conv_W_name = _find_params('conv'+str(i+1)+'f', params_names_list)[0]
        conv_b_name = _find_params('conv'+str(i+1)+'b', params_names_list)[0]
        print '\t\tCONV: setting '+conv_W_name+' '+conv_b_name
        print '\t\tCONV: stride '+str(_conv_stride[i])+', filter-group '+str(_filtergroup_yn[i])
        conv_W = params_values_list[params_names_list.index(conv_W_name)]
        conv_b = params_values_list[params_names_list.index(conv_b_name)]
        # batchnorm
        if _bnorm_yn[i]:
            bn_beta_name = _find_params('bn'+str(i+1)+'b', params_names_list)[0]
            bn_gamma_name = _find_params('bn'+str(i+1)+'m', params_names_list)[0]
            bn_moments_name = _find_params('bn'+str(i+1)+'x', params_names_list)[0]
            print '\t\tBNORM: setting '+bn_beta_name+' '+bn_gamma_name+' '+bn_moments_name
            bn_beta = params_values_list[params_names_list.index(bn_beta_name)]
            bn_gamma = params_values_list[params_names_list.index(bn_gamma_name)]
            bn_moments = params_values_list[params_names_list.index(bn_moments_name)]
            bn_moving_mean = bn_moments[:,0]
            bn_moving_variance = bn_moments[:,1]**2 # saved as std in matconvnet
        else:
            bn_beta = bn_gamma = bn_moving_mean = bn_moving_variance = []
        
        # set up conv "block" with bnorm and activation 
        net_x = set_convolutional(net_x, conv_W, np.swapaxes(conv_b,0,1), _conv_stride[i], \
                            bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance, \
                            filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
                            scope='conv'+str(i+1), reuse=False)
        
        # notice reuse=True for Siamese parameters sharing
        net_z = set_convolutional(net_z, conv_W, np.swapaxes(conv_b,0,1), _conv_stride[i], \
                            bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance, \
                            filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
                            scope='conv'+str(i+1), reuse=True)    
        
        # add max pool if required
        if _pool_stride[i]>0:
            print '\t\tMAX-POOL: size '+str(_pool_sz)+ ' and stride '+str(_pool_stride[i])
            net_x = tf.nn.max_pool(net_x, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))
            net_z = tf.nn.max_pool(net_z, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))
    
    return net_z, net_x