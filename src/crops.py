from __future__ import division
import numpy as np
import tensorflow as tf

def pad_frame(im, frame_sz, pos_x, pos_y, patch_sz, avg_chan):
	xleft_pad = tf.maximum(0, tf.cast(-tf.round(pos_x-patch_sz/2), tf.int32))
	ytop_pad = tf.maximum(0, tf.cast(-tf.round(pos_y-patch_sz/2), tf.int32))
	xright_pad = tf.maximum(0, tf.cast(tf.round(pos_x+patch_sz/2)-frame_sz[1], tf.int32))
	ybottom_pad = tf.maximum(0, tf.cast(tf.round(pos_y+patch_sz/2)-frame_sz[0], tf.int32))
	npad = tf.reduce_max([xleft_pad,ytop_pad,xright_pad,ybottom_pad])
	paddings = [[npad,npad],[npad,npad],[0,0]]
	im_padded = im - avg_chan
	im_padded = tf.pad(im_padded, paddings, mode='CONSTANT')
	im_padded = im_padded + avg_chan
	return im_padded, npad

def extract_crops_z(im, npad, pos_x, pos_y, sz_src, sz_dst):
	npad = tf.cast(npad, tf.float64)
	c = sz_src/2
	# get top-right corner of bbox and consider padding
	tr_x = tf.round(pos_x+npad-c)
	tr_y = tf.round(pos_y+npad-c)
	crop = tf.image.crop_to_bounding_box(im, tf.cast(tr_y,tf.int32), tf.cast(tr_x, tf.int32), tf.cast(sz_src, tf.int32), tf.cast(sz_src, tf.int32))
	crop = tf.image.resize_images(crop, [sz_dst,sz_dst], method=tf.image.ResizeMethod.BILINEAR)
	# crops = tf.stack([crop, crop, crop])
	crops = tf.expand_dims(crop, axis=0)
	return crops

def extract_crops_x(im, npad, pos_x, pos_y, sz_src0, sz_src1, sz_src2, sz_dst):
	npad = tf.cast(npad, tf.float64)
	# take center of the biggest scaled source patch
	c = sz_src2/2
	# get top-right corner of bbox and consider padding
	tr_x = tf.round(pos_x+npad-c)
	tr_y = tf.round(pos_y+npad-c)
	search_area = tf.image.crop_to_bounding_box(im, tf.cast(tr_y,tf.int32), tf.cast(tr_x, tf.int32), tf.cast(sz_src2, tf.int32), tf.cast(sz_src2, tf.int32))
	offset_s0 = (sz_src2-sz_src0)/2
	offset_s1 = (sz_src2-sz_src1)/2
	crop_s0 = tf.image.crop_to_bounding_box(search_area, tf.cast(offset_s0,tf.int32), tf.cast(offset_s0,tf.int32), tf.cast(sz_src0,tf.int32), tf.cast(sz_src0,tf.int32))
	crop_s0 = tf.image.resize_images(crop_s0, [sz_dst,sz_dst], method=tf.image.ResizeMethod.BILINEAR)
	crop_s1 = tf.image.crop_to_bounding_box(search_area, tf.cast(offset_s1,tf.int32), tf.cast(offset_s1,tf.int32), tf.cast(sz_src1,tf.int32), tf.cast(sz_src1,tf.int32))
	crop_s1 = tf.image.resize_images(crop_s1, [sz_dst,sz_dst], method=tf.image.ResizeMethod.BILINEAR)
	crop_s2 = tf.image.resize_images(search_area, [sz_dst,sz_dst], method=tf.image.ResizeMethod.BILINEAR)
	crops = tf.stack([crop_s0, crop_s1, crop_s2])
	return crops

	# Can't manage to use tf.crop_and_resize, which would be ideal!
	# im:  A 4-D tensor of shape [batch, image_height, image_width, depth]
	# boxes: the i-th row of the tensor specifies the coordinates of a box in the box_ind[i] image and is specified in normalized coordinates [y1, x1, y2, x2]
	# box_ind: specify image to which each box refers to
	# crop = tf.image.crop_and_resize(im, boxes, box_ind, sz_dst)

	
