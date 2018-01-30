from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters


@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)

if __name__ == "__main__":
    import sql_model, plot_utils
    import rdp
    import matplotlib.pyplot as plt
    from skimage import measure
    from skimage.filters import roberts, sobel, scharr, prewitt
    import numpy as np

    img_name = 'australian_butterfly'
    K = 150
    target_image = sql_model.get_target_image(img_name, K)
    target_pixels = target_image.pixels
    target_labels = target_image.labels
    target_labels = target_labels.astype(np.float64)
    #edge_sobel = scharr(target_labels)
    #plot_utils.display_one_patch(target_pixels, target_labels, 3)
    contours = measure.find_contours(target_pixels, 100)
    for n, contour in enumerate(contours):
        contour = rdp.rdp(contour, epsilon=5.0)
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.show()
    #plt.imshow(edge_sobel)
    #plt.show()
    #plot_utils.display_all_patches(target_pixels, target_labels)

