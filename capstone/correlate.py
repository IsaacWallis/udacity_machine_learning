"""
Uses scikit correlate function to generate a heatmap.
"""

from scipy.ndimage.filters import generic_filter
import sql_model
import numpy as np


def kernel(buffer, weights):
    diff = buffer - weights  # np.subtract(buffer, weights)
    squared = diff * diff
    return np.sum(squared)


if __name__ == "__main__":
    import patch_search
    img_name = 'australian_butterfly'
    K = 150
    target_image = sql_model.get_target_image(img_name, K)
    target_pixels = target_image.pixels
    target_labels = target_image.labels
    label = 0
    fp = target_labels == label
    fpx, fpy = np.where(fp)
    weights = target_pixels[fpx, fpy, :]

    mask = fp[np.min(fpx): np.max(fpx) + 1, np.min(fpy): np.max(fpy) + 1]
    mask = np.stack([mask, mask, mask], axis=2)

    print weights.shape
    print np.sum(mask)
    heatmap = generic_filter(target_pixels, kernel, footprint=mask, extra_arguments=(weights.flatten(),))
    heatmap = np.sqrt(heatmap)
    patch_search.plot_heatmap(heatmap)
    #print heatmap

    # q = np.arange(16)
    # q = q.reshape((4, 4))
    # q = np.stack([q, q, q], axis=2)
    #
    # t = np.zeros((3, 3, 3))
    # t[1, 1, 1] = 1
    #
    # print q
    # heatmap = generic_filter(q, kernel, footprint=t == 1, extra_arguments=(t,), mode='constant')
    # print heatmap
