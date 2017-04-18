from scipy import ndimage, misc
import image_segment
from matplotlib import pyplot as plt
import matplotlib.patches as mpatch
import numpy as np





if __name__ == "__main__":
    img = ndimage.imread('butterfly.jpg')
    img = misc.imresize(img, size = 0.0625)
    
    search_env = img
    patch_src = img
    K = 50
    labels = image_segment.segment(img, K)

    label = 40
    x_ndx, y_ndx = np.where(labels == label)
    bbox = np.min(x_ndx), np.max(x_ndx), np.min(y_ndx), np.max(y_ndx)
    modded = np.copy(img)
    modded[labels != label] = [0,0,0]
    weights = modded[bbox[0] : bbox[1] , bbox[2] : bbox[3]] / 255.

    output = ndimage.filters.convolve(img / 255., weights, mode='constant',cval=0.0)

    fig1 = plt.figure(figsize=(5, 5))
    ax1 = fig1.add_subplot(211)
    output = np.dot(output[...,:3], [0.299, 0.587, 0.114]) # greyscale
    ax1.imshow(output, alpha = 1.0)

    ax2 = fig1.add_subplot(212)
    ax2.imshow(img, alpha = 0.5)

    ax2.scatter(y_ndx,
                x_ndx,
                c = img[x_ndx, y_ndx] / 255.,
                alpha = 0.5,
                marker = ".",
                s = 5)
    plt.show()
