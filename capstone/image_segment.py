"""
Utility module with function for segmenting images into contiguous patches, and displaying
the results.
"""
import numpy as np


def segment(img, k):
    """
    Agglomerative segmenting. 
    
    :param img: An RGB image
    :param k: the number of clusters required

    :returns: A matrix of the same height and width as img. Each element is one of K labels
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.image import grid_to_graph
    import time
    print "clustering: K = %d" % k
    t0 = time.time()
    connectivity = grid_to_graph(img.shape[0], img.shape[1])
    if len(img.shape) is 3:
        x = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    elif len(img.shape) is 2:
        x = np.reshape(img, (img.shape[0] * img.shape[1], 1))
    ward = AgglomerativeClustering(n_clusters=k, linkage='ward',
                                   connectivity=connectivity)
    ward.fit(x)
    labels = np.reshape(ward.labels_, (img.shape[0], img.shape[1]))
    print("Elapsed time: ", time.time() - t0)
    print("Number of pixels: ", labels.size)
    print("Number of clusters: ", np.unique(labels).size)
    return labels


def sort_patch_indices(labels):
    max_label = np.max(labels)
    hist = np.histogram(labels, bins=max_label)
    sorted_list = np.flip(np.argsort(hist[0]), 0)
    return sorted_list


if __name__ == "__main__":
    import sql_model
    import plot_utils
    target_image = sql_model.get_target_image('small_butterfly', 50)
    plot_utils.display_all_patches(target_image.pixels, target_image.labels)
