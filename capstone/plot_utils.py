from matplotlib import pyplot as plt
import numpy as np

def display_all_patches(img, labels):
    """
    Displays clusters on a plot.

    :param img: An RGB image
    :param labels: matrix of same size as img, densely labelled.
    """
    plt.figure(figsize=(5, 5))
    for l in range(np.max(labels) + 1):
        color = np.mean(img[labels == l], axis=0) / 255.
        segment_indices = np.where(labels == l)
        plt.scatter(segment_indices[1],
                    segment_indices[0],
                    color=color,
                    alpha=.5,
                    marker=",",
                    s=5)
        #plt.contour(labels == l, contours=1,
        #            colors=[color], alpha=1., linewidths=0.1)
    plt.imshow(img, alpha=.5)
    plt.show()


def display_one_patch(img, labels, label):
    """
    Displays cluster on a plot.

    :param img: An RGB image        
    :param labels: matrix of same size as img, densely labelled.
    :param label: the label of the patch to show
    """
    plt.figure(figsize=(5, 5))
    color = np.mean(img[labels == label], axis=0) / 255.
    segment_indices = np.where(labels == label)
    plt.scatter(segment_indices[1],
                segment_indices[0],
                color=color,
                alpha=0.5,
                marker=",",
                s=5)
    plt.contour(labels == label, contours=2,
                colors=[color], alpha=1., linewidths=0.25)
    plt.imshow(img, alpha=.5)
    plt.show()