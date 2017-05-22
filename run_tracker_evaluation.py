import sys

from src.tracker import tracker

def main():
    hp = {"z_lr":0.00}
    evaluation = {"video": "vot2016_ball1"}
    run = {"visualization":0,"debug":0}
    tracker(hp, evaluation, run)

if __name__ == '__main__':
    sys.exit(main())
