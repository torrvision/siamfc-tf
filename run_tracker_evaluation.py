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
    hp = {"z_lr":0.00}
    evaluation = {"video": "all"}
    run = {"visualization":0,"debug":0}
    
    hp,evaluation,run,env,design = parse_arguments(hp, evaluation, run)

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

    # overlap = _compute_overlap(gt, bboxes)

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

# def _compute_overlap(gt, bboxes):

if __name__ == '__main__':
    sys.exit(main())
