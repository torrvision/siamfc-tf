from __future__ import division
import numpy as np
import tensorflow as tf

def pad_frame(im, frame_sz, pos, patch_sz, avg_chan):
	xleft_pad = max(0, -np.round(pos[1]-patch_sz/2))
	ytop_pad = max(0, -np.round(pos[0]-patch_sz/2))
	xright_pad = max(0, np.round(pos[1]+patch_sz/2)-frame_sz[1])
	ybottom_pad = max(0, np.round(pos[0]+patch_sz/2)-frame_sz[0])
	npad = np.int32(max(xleft_pad,ytop_pad,xright_pad,ybottom_pad))
	paddings = [[npad,npad],[npad,npad],[0,0]]
	im_padded = im - avg_chan
	im_padded = tf.pad(im_padded, paddings, mode='CONSTANT')
	im_padded = im_padded + avg_chan
	return im_padded, npad

def extract_crops(im, npad, pos, sz_src, sz_dst):
	num_scales = np.size(sz_src)
	# im = np.expand_dims(im, axis=0)
	sz_dst = np.int32(sz_dst)
		
	if num_scales==3:
		# take center of the biggest scaled source patch
		c = sz_src[-1]/2
		pos = np.int32(pos+npad-c)
		search_area = tf.image.crop_to_bounding_box(im, pos[0], pos[1], np.int32(sz_src[-1]), np.int32(sz_src[-1]))
		offset_s0 = np.int32((sz_src[-1]-sz_src[0])/2)
		offset_s1 = np.int32((sz_src[-1]-sz_src[1])/2)		
		# with tf.device("/gpu:0"):
		crop_s0 = tf.image.crop_to_bounding_box(search_area, offset_s0, offset_s0, np.int32(sz_src[0]), np.int32(sz_src[0]))
		crop_s0 = tf.image.resize_images(crop_s0, [sz_dst,sz_dst], method=tf.image.ResizeMethod.BILINEAR)
		crop_s1 = tf.image.crop_to_bounding_box(search_area, offset_s1, offset_s1, np.int32(sz_src[1]), np.int32(sz_src[1]))
		crop_s1 = tf.image.resize_images(crop_s1, [sz_dst,sz_dst], method=tf.image.ResizeMethod.BILINEAR)
		crop_s2 = tf.image.resize_images(search_area, [sz_dst,sz_dst], method=tf.image.ResizeMethod.BILINEAR)
		crops = tf.stack([crop_s0, crop_s1, crop_s2])
	else:
		if num_scales==1:
			c = sz_src/2
			pos = np.int32(pos+npad-c)
			# with tf.device("/gpu:0"):
			crop = tf.image.crop_to_bounding_box(im, pos[0], pos[1], np.int32(sz_src), np.int32(sz_src))
			crop = tf.image.resize_images(crop, [sz_dst,sz_dst], method=tf.image.ResizeMethod.BILINEAR)
			crops = tf.stack([crop, crop, crop])
		else:
			raise ValueError('Code working ony for 1 or 3 scales.')
	
	return crops

	# Can't manage to use tf.crop_and_resize, which would be ideal! box_ind is never of the approriate rank
	# im:  A 4-D tensor of shape [batch, image_height, image_width, depth]
	# boxes: the i-th row of the tensor specifies the coordinates of a box in the box_ind[i] image and is specified in normalized coordinates [y1, x1, y2, x2]
	# box_ind: specify image to which each box refers to
	# crop = tf.image.crop_and_resize(im, boxes, box_ind, sz_dst)

	
