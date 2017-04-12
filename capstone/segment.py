import numpy as np
import cv2


img = cv2.imread('butterfly.jpg')
img = cv2.resize(img, (0,0), fx = 0.25, fy = 0.25)

def kmeansproximity(img):
    K = 20
    
    rows = np.arange(img.shape[0])
    cols = np.arange(img.shape[1])
    features = np.zeros((img.shape[0], img.shape[1], 5))

    #add the RGB values to the features
    features[:,:,:3] = img#cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #Add the pixel index values to the features for proximity clustering
    tiledRowIndices = np.tile(np.arange(img.shape[0]), (img.shape[1], 1))
    tiledColIndices = np.tile(np.arange(img.shape[1]), (img.shape[0], 1))
    
    proximityImportance = 1.5
    features[:,:,3] = tiledRowIndices.T * proximityImportance
    features[:,:,4] = tiledColIndices * proximityImportance
    
    Z = features.reshape((-1,5))
    
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((features.shape))
    painted = res2[:,:,:3]
    display = cv2.addWeighted(painted, .8, img, .2, 0)
    cv2.imshow('butterfly',display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def agglom(img):
    img = cv2.resize(img, (0,0), fx = 0.25, fy = 0.25)
    K = 100
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.image import grid_to_graph
    import time
    print "clustering: K = %d" % K
    t0 = time.time()
    connectivity = grid_to_graph(img.shape[0], img.shape[1])
    print connectivity.shape
    X = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    print X.shape
    ward =  AgglomerativeClustering(n_clusters = K, linkage = 'ward',
                                    connectivity = connectivity)
    ward.fit(X)
    print ward.labels_.shape
    labels = np.reshape(ward.labels_, (img.shape[0], img.shape[1] ))
    print("Elapsed time: ", time.time() - t0)
    print("Number of pixels: ", labels.size)
    print("Number of clusters: ", np.unique(labels).size)
    #plot clusters
    display_cluster_avg(img, labels, K)
    #plot_cluster_contours(img, labels, K)

def plot_cluster_contours(img, labels, K):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=plt.cm.gray)
    for l in range(K):
        plt.contour(labels == l, contours=1,
                    colors=[plt.cm.spectral(l / float(K)), ])
    plt.xticks(())
    plt.yticks(())
    plt.show()

def display_cluster_avg(img, labels, K):
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatch
    import numpy as np
    
    plt.figure(figsize=(5, 5))

    #axes = plt.gca()
    #axes.add_patch(patch)
    for l in range(K):
        #l = 4
        color = np.mean(img[labels == l], axis = 0) / 255.
        segmentIndices =  np.where(labels == l)
        plt.scatter(segmentIndices[1], segmentIndices[0], color = color, alpha = 0.5, marker = ",", s = 5)

        plt.contour(labels == l, contours=1,
                     colors=[color], alpha = 1., linewidths = 0.1)
    plt.imshow(img, alpha = .5)
    plt.show()
    
agglom(img)
