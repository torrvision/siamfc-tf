import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def show_frame(frame, fig_n):
    fig = plt.figure(fig_n)
    ax = fig.add_subplot(111)
    ax.imshow(np.uint8(frame))
    plt.ion()
    plt.show()
    plt.pause(0.001)

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
    plt.pause(1)