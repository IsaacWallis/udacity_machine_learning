"""
Utility module with function for segmenting images into contiguous patches, and displaying
the results.
"""
import numpy as np
import file_handling


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


def display_all_patches(img, labels):
    """
    Displays clusters on a plot.

    :param img: An RGB image
    :param labels: matrix of same size as img, densely labelled.
    """
    from matplotlib import pyplot as plt
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
        plt.contour(labels == l, contours=1,
                    colors=[color], alpha=1., linewidths=0.1)
    plt.imshow(img, alpha=.5)
    plt.show()


def display_one_patch(img, labels, label):
    """
    Displays cluster on a plot.

    :param img: An RGB image        
    :param labels: matrix of same size as img, densely labelled.
    :param label: the label of the patch to show
    """
    from matplotlib import pyplot as plt
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


def get_segmented_image(name, k):
    """
    Gets a segmented image from file if the file exists. If not, segments the image
    and writes a segment file for next time.
    
    :param name: The name of the image file to segment, without suffix. Image file must exist. 
    :param k: number of segments desired
    :returns: A dictionary with the data from a segmented image
    """
    img = file_handling.get_target_image("small_butterfly")
    if file_handling.project_exists(name):
        print "opening project"
        segment_data = file_handling.read_segment_file(name)
        pix = segment_data["pixels"]
        labels = segment_data["labels"]
        if k != (np.max(labels) + 1) or img.shape != pix.shape:
            print "segment file parameters don't match, remaking."
            print "K:", k, np.max(labels) + 1
            print "shape:", img.shape, pix.shape
            file_handling.remove_segment_file(name)
        else:
            return segment_data
    labels = segment(img, k)
    segment_data = init_project_dict(img, labels)
    file_handling.write_segment_file(name, segment_data)
    return segment_data


def init_project_dict(pixels, labels):
    segment_data = {
        "pixels": pixels,
        "labels": labels
    }
    max_label = np.max(labels)
    hist = np.histogram(labels, bins=max_label)
    order = np.flip(np.argsort(hist[0]), 0)
    for i in order:
        key = "patch_%i" % i
        segment_data[key] = {}
        segment_data[key]["size"] = hist[0][i]
        segment_data[key]["visits"] = {}
    return segment_data


if __name__ == "__main__":
    segData = get_segmented_image('small_butterfly', 50)
    display_all_patches(segData["pixels"], segData["labels"])
