import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2


def show_frame(frame, bbox, fig_n):
    #Adjust Color Channels of frame
    frame_adjusted = np.ndarray(shape=(720,1280,3), dtype=np.dtype(np.uint8))
    frame_adjusted[:,:,:] = frame[:,:,2::-1]
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame_adjusted, (x, y), (x+w,y+h), (0,0,255), 4, 8, 0)
    cv2.imshow('image',frame_adjusted)
    cv2.waitKey(1)


def show_crops(crops, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(np.uint8(crops[0,:,:,:]))
    ax2.imshow(np.uint8(crops[1,:,:,:]))
    ax3.imshow(np.uint8(crops[2,:,:,:]))
    plt.ion()
    plt.show()
    plt.pause(0.001)


def show_scores(scores, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(scores[0,:,:], interpolation='none', cmap='hot')
    ax2.imshow(scores[1,:,:], interpolation='none', cmap='hot')
    ax3.imshow(scores[2,:,:], interpolation='none', cmap='hot')
    plt.ion()
    plt.show()
    plt.pause(0.001)
