"""
Utility module with function for segmenting images into contiguous patches, and displaying
the results.
"""
import numpy as np
import pickle

segmented_dir = "./segmented/"

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
    if len(img.shape) is 3:
        X = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    elif len(img.shape) is 2:
        X = np.reshape(img, (img.shape[0] * img.shape[1], 1))
    ward =  AgglomerativeClustering(n_clusters = K, linkage = 'ward',
                                    connectivity = connectivity)
    ward.fit(X)
    labels = np.reshape(ward.labels_, (img.shape[0], img.shape[1] ))
    print("Elapsed time: ", time.time() - t0)
    print("Number of pixels: ", labels.size)
    print("Number of clusters: ", np.unique(labels).size)
    return labels

def get_segmented_img(name, K):
    """
    Gets a segmented image from file if the file exists. If not, segments the image
    and writes a segment file for next time.
    
    :param name: The name of the image file to segment, without suffix. Image file must exist. 
    :param K: number of segments desired
    :returns: A dictionary with the data from a segmented image
    """
    import os.path
    from scipy import ndimage, misc
    img = ndimage.imread(name + '.jpg')    
    if os.path.isfile(segmented_dir + name + '.segmented'):
        segment_data = read_segment_file(segmented_dir + name + '.segmented')
        pix = segment_data["pixels"]
        labels = segment_data["labels"]
        if K != (np.max(labels) + 1) or img.shape != pix.shape:
            print "segment file parameters don't match, remaking."
            print "K:",K, np.max(labels) + 1
            print "shape:",img.shape, pix.shape
        else:
            return segment_data            
    labels = segment(img, K)
    segment_data = {
        "pixels" : img,
        "labels" : labels
        }
    write_segment_file(segmented_dir + name + '.segmented', segment_data)
    return segment_data
        
def write_segment_file(name, data):
    output = open(name, 'ab+')
    pickle.dump(data, output)
    output.close()

def read_segment_file(name):
    output = open(name, 'rb')
    print "o", output
    data = pickle.load(output)
    output.close()
    return data
    
def display_all_patches(img, labels, K):
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

def display_one_patch(img, labels, label):
    """
    Displays cluster on a plot.

    :param img: An RGB image
    :param label: the label of the patch to show
    :returns: A matrix of the same height and width as img. Each element is one of K labels
    """
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatch
    plt.figure(figsize=(5, 5))
    color = np.mean(img[labels == label], axis = 0) / 255.
    segmentIndices =  np.where(labels == label)
    plt.scatter(segmentIndices[1],
                segmentIndices[0],
                color = color,
                alpha = 0.5,
                marker = ",",
                s = 5)
    plt.contour(labels == label, contours=2,
                colors=[color], alpha = 1., linewidths = 0.25)
    plt.imshow(img, alpha = .5)
    plt.show()
        
def segment_as_patch(img, labels, patch_index):
    px = img[labels == patch_index]
    x_indices, y_indices = np.where(labels == patch_index)
    patch = np.zeros((px.shape[0], 5))
    patch[:, :3] = px
    patch[:, 3] = x_indices
    patch[:, 4] = y_indices
    return patch

def image_as_patch(img):
    x_indices, y_indices = np.indices((img.shape[0], img.shape[1]))
    patch = np.zeros((img.shape[0], img.shape[1], 5))
    patch[:, :, :3] = img
    patch[:, :, 3] = x_indices
    patch[:, :, 4] = y_indices
    patch = np.reshape(patch, (patch.shape[0] * patch.shape[1], 5))
    return patch

def patch_as_image(patch):
    size_x = np.max(patch[:, 3]) - np.min(patch[:, 3])
    size_y = np.max(patch[:, 4]) - np.min(patch[:, 4])
    patch = np.reshape(patch, (size_x + 1,  size_y + 1, 5))
    return patch[:, :, :3]
    
if __name__ == "__main__":
    from scipy import ndimage, misc
    img = ndimage.imread('butterfly.jpg')
    img = misc.imresize(img, size = 0.0625)
    K = 50
    labels = segment(img, K)
    display_all_patches(img, labels, K)
