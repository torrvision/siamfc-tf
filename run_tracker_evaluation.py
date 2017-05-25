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
	# avoid printing TF debugging information
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	# TODO: this will be passed to the main function
	hp = {}
	evaluation = {"video": "all"}
	run = {"visualization":0,"debug":0}

	# read all default parameters and overwrite ones defined by user
	hp,evaluation,run,env,design = parse_arguments(hp, evaluation, run)
	final_score_sz = int(hp.response_up * design.score_sz)
	# build TF graph once for all
	filename, image, templates_z, scores = siam.build_tracking_graph(final_score_sz, design, env)	

	# iterate through all videos of evaluation.dataset
	if evaluation.video=='all':
		dataset_folder = os.path.join(env.root_dataset, evaluation.dataset)
		videos_list = [v for v in os.listdir(dataset_folder)]
		videos_list.sort()
		nv = np.size(videos_list)
		speed = np.zeros(nv)
		ious = np.zeros(nv)
		lengths = np.zeros(nv)
		for i in range(nv):
			gt, frame_name_list, frame_sz, pos_x, pos_y, target_w, target_h  = _init_video(env, evaluation, videos_list[i])
			bboxes, speed[i] = tracker(hp, evaluation, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz, filename, image, templates_z, scores)
			lengths[i], ious[i] = _compile_results(gt, bboxes, videos_list[i])
			print str(i)+' -- '+videos_list[i]+' -- IOU: '+("%.2f" % ious[i])+' -- Speed: '+("%.2f" % speed[i])+' --'
			print

		tot_frames = np.sum(lengths)
		mean_iou = np.sum(ious*lengths)/tot_frames
		mean_speed = np.sum(speed*lengths)/tot_frames
		print '-- Overall stats (averaged per frame) on '+str(nv)+' videos ('+str(tot_frames)+' frames) --'
		print '-- IOU: '+("%.2f" % mean_iou)+' -- Speed: '+("%.2f" % mean_speed)+ ' --'
		print

	else:
		gt, frame_name_list, frame_sz, pos_x, pos_y, target_w, target_h = _init_video(env, evaluation, evaluation.video)
		bboxes, speed = tracker(hp, evaluation, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz, filename, image, templates_z, scores)
		_, iou = _compile_results(gt, bboxes, evaluation.video)
		print evaluation.video+' -- IOU: '+("%.2f" % iou)+' -- Speed: '+("%.2f" % speed)+' --'

def _compile_results(gt, bboxes, video):
	l = np.size(bboxes,0)
	gt4 = np.zeros((l, 4))
	new_ious = np.zeros(l)
	# np.savetxt('out/'+video+'.bboxes', bboxes, delimiter=',')
	# np.savetxt('out/'+video+'.gt', gt, delimiter=',')
	for j in range(l):
		gt4[j, :] = region_to_bbox(gt[j, :], center=False)
		new_ious[j] = _compute_iou(bboxes[j,:], gt4[j,:])

	iou = np.mean(new_ious)*100

	return l, iou

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
	# determine the (x, y)-coordinates of the intersection rectangle	
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
	yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
 
 	if xA<xB and yA<yB:
		# compute the area of intersection rectangle
		interArea = (xB - xA) * (yB - yA) 
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = boxA[2] * boxA[3]
		boxBArea = boxB[2] * boxB[3]
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
	else:
		iou = 0

	assert iou >=0
	assert iou <= 1.01

	return iou

if __name__ == '__main__':
    sys.exit(main())
