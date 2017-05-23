import sys
import os
import numpy as np
from PIL import Image

import src.siamese as siam
from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
from src.pprint_params import pprint_params

def main():

	# TODO: this will be passed to the main function
	hp = {"z_lr":0.006}
	evaluation = {"video": "vot2016_helicopter"}
	run = {"visualization":0,"debug":0}

	# read all default parameters and overwrite ones defined by user
	hp,evaluation,run,env,design = parse_arguments(hp, evaluation, run)

	# iterate through all videos of evaluation.dataset
	if evaluation.video=='all':
		dataset_folder = os.path.join(env.root_dataset, evaluation.dataset)
		videos_list = [v for v in os.listdir(dataset_folder)]
		videos_list.sort()
		for i in range(np.size(videos_list)):
			gt, frame_name_list, frame_sz, pos_x, pos_y, target_w, target_h  = _init_video(env, evaluation, videos_list[i])
			bboxes, speed = tracker(hp, evaluation, run, env, design, frame_name_list, frame_sz, pos_x, pos_y, target_w, target_h)
	else:
		gt, frame_name_list, frame_sz, pos_x, pos_y, target_w, target_h = _init_video(env, evaluation, evaluation.video)
		bboxes, speed = tracker(hp, evaluation, run, env, design, frame_name_list, frame_sz, pos_x, pos_y, target_w, target_h)

	n_bboxes = np.shape(bboxes)[0]
	ious = np.zeros(n_bboxes)
	for i in range(n_bboxes):
		ious[i] = _compute_iou(bboxes[i,:], gt[i])

	print ious
	print 'Average IOU: '+str(np.mean(ious))
    	
def _init_video(env, evaluation, video):
	    video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
	    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
	    frame_name_list = [os.path.join(env.root_dataset, evaluation.dataset, video, '') + s for s in frame_name_list]
	    frame_name_list.sort()    
	    with Image.open(frame_name_list[0]) as img:
	        frame_sz = np.asarray(img.size)
	        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

	    # read the initialization from ground truth
	    gt_file = os.path.join(video_folder, 'groundtruth.txt')
	    gt = np.genfromtxt(gt_file, delimiter=',')
	    assert len(gt) == len(frame_name_list), ('Number of frames and number of GT lines should be equal.')
	    ## tracker's state initializations, bbox is in format <cx,cy,w,h>
	    pos_x, pos_y, target_w, target_h = region_to_bbox(gt[evaluation.start_frame])
	    return gt, frame_name_list, frame_sz, pos_x, pos_y, target_w, target_h

def _compute_iou(boxA, boxB):
	boxA = region_to_bbox(boxA, center=False)
	boxB = region_to_bbox(boxB, center=False)
	# determine the (x, y)-coordinates of the intersection rectangle	
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
	yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
 
	# compute the area of intersection rectangle
	interArea = (xB - xA) * (yB - yA)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = boxA[2] * boxA[3]
	boxBArea = boxB[2] * boxB[3]
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = max(0, interArea / float(boxAArea + boxBArea - interArea))
 
	# return the intersection over union value
	return iou

if __name__ == '__main__':
    sys.exit(main())
