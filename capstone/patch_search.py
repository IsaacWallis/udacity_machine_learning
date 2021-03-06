import image_segment
import numpy as np
from scipy import ndimage, misc

"""
Provides exhaustive, gradient descent, and RL based patch searching functions.
"""    

def similarity(patchA, patchB):
    """
        A similarity metric that compares segments of the same shape, pixel-by-pixel.
        
        :param patchA: A set of pixels
        :param patchB:
        :returns: 0 to 1 float, where 1 means the patches are identical
        :raises AssertionError: if the patch shapes are not the same
        """
    assert patchA.shape == patchB.shape
    segment = patchA / 255.
    otherSegment = patchB / 255.
    error = np.subtract(segment, otherSegment)
    squVoxelError = error * error
    pixelError = np.sum(squVoxelError, axis = 1) / 3.
    totalError = np.sum(pixelError) / len(pixelError)
    return 1.0 - totalError
        

def grid_search(env_pixels, patch_pixels, patch_indices):
    """
    Performs a grid search to find the patch in the environment which best matches the search patch. Slow.

    :param env_pixels: The pixels of the image to search.
    :param patch_pixels: The pixels of the patch used as the search key.
    :param patch_indices: The indices of the patch translated to origin (used to define patch shape).
    :returns: An array of the same size as env, can be plotted as a heat map.
    """
    import time

    x_ndcs = patch_indices[0]
    y_ndcs = patch_indices[1]
    heatmap = np.zeros((env_pixels.shape[0] - np.max(x_ndcs), env_pixels.shape[1] - np.max(y_ndcs)))
    t0 = time.time()

    print heatmap.shape, env_pixels.shape, np.max(x_ndcs)
    for i in range(heatmap.shape[0] ):
        for j in range(heatmap.shape[1] ):
            trans_x = x_ndcs + i
            trans_y = y_ndcs + j
            pixels_to_check = env_pixels[trans_x, trans_y]
            sim = similarity(patch_pixels, pixels_to_check) #TODO scale this? 
            heatmap[i,j] = sim            
    print("Elapsed time: ", time.time() - t0)
    return heatmap

def plot_heatmap(heatmap):
    """
    Plots a heatmap, and draws an X on the location of the max value.

    :param env_pixels: The environment pixels. Displayed in a separate plot.
    :param patch_pixels: Overlayed on top of the environment pixels.
    :param heatmap: A heatmap generated by search function such as grid_search or gradient_descent.
    """
    from matplotlib import pyplot as plt
    fig1 = plt.figure(figsize=(5, 5))
    ax1 = fig1.add_subplot(211)
    ax1.imshow(heatmap, alpha = 1.0)
    
    max_ndx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    ax1.scatter(max_ndx[1], max_ndx[0], c='red', marker="x")
    plt.show()    

def display_patch(env_pixels, patch_pixels, patch_indices, state):
    """
    Displays the patch overlayed on the environment in the spot indicated by state.

    :param env_pixels: The environment that was searched
    :param patch_pixels: The patch that was used as search key.
    :param patch_indices: The indices of the patch, translated to origin min(0,0)
    :param state: The location of the patch within the environment. Should be set by a search function.
    """
    from matplotlib import pyplot as plt
    
    plt.figure(figsize=(5, 5))
    plt.imshow(env_pixels, alpha = .5)
    x = patch_indices[0]
    y = patch_indices[1]
    trans_x = x + state[0]
    trans_y = y + state[1]
    
    plt.scatter(trans_y,
                trans_x,
                c = patch_pixels / 255.,
                alpha = 1.0,
                marker = ".",
                s = 5)
    plt.show()

if __name__ == "__main__":
    import sql_model, file_handling
    img_name = 'small_butterfly'
    K = 150
    seg_data = sql_model.make_project_file(img_name, K)
    labels = seg_data["labels"]
    pixels = seg_data["pixels"]

    LABEL = 0
    patch_indices = np.where(labels == LABEL)
    patch_pixels = pixels[patch_indices]    
    patch_indices = (patch_indices[0] - np.min(patch_indices[0]),
                         patch_indices[1] - np.min(patch_indices[1]))

    env_pixels = file_handling.get_target_image('small_butterfly')
    heatmap = grid_search(env_pixels, patch_pixels, patch_indices)
    plot_heatmap(heatmap)
    max_ndx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    display_patch(env_pixels, patch_pixels, patch_indices, max_ndx)
