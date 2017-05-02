import numpy as np

def extract_crop(im, pos, sz_src, sz_dst, chan_avg):

	pad_l, pad_t, pad_r, pad_b = check_out_of_frame(np.asarray(np.shape(im)), src_sz, pos)


def check_out_of_frame(im_shape, sz_src, pos):
	im_shape = im_shape[0:2]+1



