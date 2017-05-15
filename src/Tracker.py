from __future__ import division
import numpy as np
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import os.path
from src.BBox import BBox

class Tracker:

	def __init__(self, bbox, design):
		self.bbox = BBox(bbox[0], bbox[1], bbox[2], bbox[3])
		self.z_sz, self.x_sz = self.get_crops_sz(design.context, design.exemplar_sz, design.search_sz)	

	def get_crops_sz(self, context_amount, exemplar_sz, search_sz):
		context = context_amount*sum(self.bbox.target_sz)
		crop_sz = self.bbox.target_sz + context
		z_sz = np.sqrt(np.prod(crop_sz))
		x_sz = search_sz/exemplar_sz * z_sz	
		return z_sz, x_sz




