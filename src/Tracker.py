import numpy as np
import sys
sys.path.append('../')
from src.BBox import BBox

class Tracker:

	def __init__(self, bbox, frame, design):
		self.bbox = BBox(bbox[0], bbox[1], bbox[2], bbox[3])
		self.frame = frame # current frame being processed
		self.z_sz, self.x_sz = self.get_crops_sz(design.context, design.search_sz/design.exemplar_sz)

	def get_crops_sz(self, context_amount, ratio):
		context = context_amount*(self.bbox.w+self.bbox.h)
		crop_w = self.bbox.w + context
		crop_h = self.bbox.h + context
		sqrt_a = np.sqrt(crop_w*crop_h)
		z_sz, x_sz =  sqrt_a * np.array((1, ratio))
		return z_sz, x_sz


