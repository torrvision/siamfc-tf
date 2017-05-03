from __future__ import division
import numpy as np
import tensorflow as tf

def pad_frame(im, pos, patch_sz, avg_chan):
	frame_sz = np.asarray(np.shape(im))[0:2]
	xleft_pad = max(0, -np.round(pos[1]-patch_sz/2))
	ytop_pad = max(0, -np.round(pos[0]-patch_sz/2))
	xright_pad = max(0, np.round(pos[1]+patch_sz/2)-frame_sz[1])
	ybottom_pad = max(0, np.round(pos[0]+patch_sz/2)-frame_sz[0])
	npad = np.int16(max(xleft_pad,ytop_pad,xright_pad,ybottom_pad))
	# pad image uniformly all around, each channel is padded independently
	imR = np.expand_dims(np.pad(im[:,:,0], npad, mode='constant', constant_values=np.round(avg_chan[0])), 2)
	imG = np.expand_dims(np.pad(im[:,:,1], npad, mode='constant', constant_values=np.round(avg_chan[1])), 2)
	imB = np.expand_dims(np.pad(im[:,:,2], npad, mode='constant', constant_values=np.round(avg_chan[2])), 2)
	im_padded = np.concatenate((imR,imG,imB),axis=2)
	return im_padded, npad

def extract_crops(im, npad, pos, sz_src, sz_dst, pyramid_sz):
	sz_src = np.int32(np.round(sz_src))
	c = sz_src/2
	# we need to consider that the frame has been padded along both axes
	pos = np.int32(pos+npad)
	if pyramid_sz>1:
		max_target_side = sz_src[-1]
		min_target_side = sz_src[0]
		search_side = np.round(sz_dst*max_target_side/min_target_side)
		crop = []
	else:
		search_side = np.round(sz_dst)
		boxes = np.expand_dims((pos[0]-c, pos[1]-c, pos[0]+c, pos[1]+c), axis=0)
		box_ind = np.empty([0,1])
		box_ind[:,0]=0
		crop = tf.image.crop_to_bounding_box(im, pos[0], pos[1], sz_src, sz_src)		
	# Can't use this, which would be ideal! box_ind is never of the approriate rank
	# im:  A 4-D tensor of shape [batch, image_height, image_width, depth]
	# boxes: the i-th row of the tensor specifies the coordinates of a box in the box_ind[i] image and is specified in normalized coordinates [y1, x1, y2, x2]
	# box_ind: specify image to which each box refers to
	# crop = tf.image.crop_and_resize(im, boxes, box_ind, sz_dst)


	return crop
