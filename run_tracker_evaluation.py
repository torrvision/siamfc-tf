from __future__ import division, print_function
import argparse
import sys
import os

import numpy as np
from PIL import Image, ImageDraw

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

pos_x, pos_y, target_w, target_h = args.x + args.w/2, args.y + args.h/2, args.w, args.h

def main():
    # avoid printing TF debugging information
#    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

    # print("Written file names")
    # for f, coords, index in zip(frame_names, bboxes, range(1000)):
    #     img = Image.open(f)
    #     img_d = ImageDraw.Draw(img)
    #     rect = ((coords[0], coords[1]), (coords[2], coords[3]))
    #     img_d.rectangle( rect )

    #     if index % 5 == 0:
    #         print(coords)
    #         img.show()
    #         raw_input("Continue?")

    print(evaluation.video + ' -- Speed: ' + "%.2f" % speed + ' --')



if __name__ == '__main__':
    sys.exit(main())
