from __future__ import division, print_function
import argparse
import sys
import os

import numpy as np

from src.parse_arguments import parse_arguments
from src.tracker import tracker
import src.siamese as siam

parser = argparse.ArgumentParser()
parser.add_argument("--x", type=int, required=True)
parser.add_argument("--y", type=int, required=True)
parser.add_argument("--w", type=int, required=True)
parser.add_argument("--h", type=int, required=True)
parser.add_argument("-s", "--source", help="dir of frame .jpgs",
                    type=str, required=True)
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

    frame_names = [os.path.join(args.source, f)
                   for f in os.listdir(args.source) if f.endswith(".jpg")]

    frame_names.sort()

    bboxes, speed = tracker(hp, run, design, frame_names,
                            pos_x, pos_y, target_w, target_h, final_score_sz,
                            filename, image, templates_z, scores,
                            start_frame=0)

    np.savetxt("bboxes", bboxes)
    with open("filenames", "w") as framef:
        framef.writelines("\n".join(frame_names))

    print(evaluation.video + ' -- Speed: ' + "%.2f" % speed + ' --')



if __name__ == '__main__':
    sys.exit(main())
