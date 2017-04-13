"""
Utility module with function for segmenting images into contiguous patches, and displaying
the results.
"""
import numpy as np
import cv2

def segment(img, K):
    """
    Agglomerative segmenting. 
    
    :param img: An RGB image
    :param K: the number of clusters required

    :returns: A matrix of the same height and width as img. Each element is one of K labels
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.image import grid_to_graph
    import time
    print "clustering: K = %d" % K
    t0 = time.time()
    connectivity = grid_to_graph(img.shape[0], img.shape[1])
    X = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    ward =  AgglomerativeClustering(n_clusters = K, linkage = 'ward',
                                    connectivity = connectivity)
    ward.fit(X)
    labels = np.reshape(ward.labels_, (img.shape[0], img.shape[1] ))
    print("Elapsed time: ", time.time() - t0)
    print("Number of pixels: ", labels.size)
    print("Number of clusters: ", np.unique(labels).size)
    return labels

def display_cluster_avg(img, labels, K):
    """
    Displays clusters on a plot.

    :param img: An RGB image
    :param K: the number of clusters required
    :returns: A matrix of the same height and width as img. Each element is one of K labels
    """
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatch
    plt.figure(figsize=(5, 5))
    for l in range(K):
        color = np.mean(img[labels == l], axis = 0) / 255.
        segmentIndices =  np.where(labels == l)
        plt.scatter(segmentIndices[1],
                    segmentIndices[0],
                    color = color,
                    alpha = 0.5,
                    marker = ",",
                    s = 5)
        plt.contour(labels == l, contours=1,
                    colors=[color], alpha = 1., linewidths = 0.1)
    plt.imshow(img, alpha = .5)
    plt.show()

if __name__ == "__main__":
    img = cv2.imread('butterfly.jpg')
    img = cv2.resize(img, (0,0), fx = 0.0625, fy = 0.0625)
    K = 50
    labels = segment(img, K)
    display_cluster_avg(img, labels, K)
