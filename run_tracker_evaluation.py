from __future__ import division, print_function
import argparse
import sys
import os

from PIL import Image
import numpy as np

from src.parse_arguments import parse_arguments
from src.tracker import tracker
import src.siamese as siam

parser = argparse.ArgumentParser()
parser.add_argument("--x", type=int, required=True)
parser.add_argument("--y", type=int, required=True)
parser.add_argument("--w", type=int, required=True)
parser.add_argument("--h", type=int, required=True)
args = parser.parse_args()

pos_x, pos_y, target_w, target_h = args.x, args.y, args.w, args.h


def main():
    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    hp, evaluation, run, env, design = parse_arguments()
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1

    # build TF graph once for all
    filename, image, templates_z, scores = siam.build_tracking_graph(
        final_score_sz, design, env)

    frame_name_list = _init_video(env, evaluation)

    bboxes, speed = tracker(hp, run, design, frame_name_list,
                            pos_x, pos_y, target_w, target_h, final_score_sz,
                            filename, image, templates_z, scores,
                            start_frame=0)

    np.savetxt("bboxes", bboxes)
    with open("filenames", "w") as framef:
        framef.writelines(frame_name_list)

    print(evaluation.video + ' -- Speed: ' + "%.2f" % speed + ' --')


def _init_video(env, evaluation):
    """
    Take an environment config and and evaluation config and return a list of
    filenames of frames.
    """
    video = evaluation.video
    video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)

    frame_names = [os.path.join(env.root_dataset, evaluation.dataset, video, f)
                   for f in os.listdir(video_folder) if f.endswith(".jpg")]

    frame_names.sort()
    return frame_names


if __name__ == '__main__':
    sys.exit(main())
