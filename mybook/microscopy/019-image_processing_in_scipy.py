# https://www.youtube.com/watch?v=s_hDL2fGvow&t=

"""
#Image processing using Scipy
Scipy is a python library that is part of numpy stack. 
It contains modules for linear algebra, FFT, signal processing and
image processing. Not designed for image processing but has a few tools
"""

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import skimage

# 讀檔 ----------
# scipy 沒提供，就用任何一個，讀完是 numpy array 的 function 就好
img = skimage.io.imread("images/monkey.jpg")
print(type(img))  # numpy array
img.shape  # (330, 220, 3)
print(img.min(), img.max())  # 0 255 -> uint8

# 轉成灰階
grey_img = skimage.io.imread("images/monkey.jpg", as_gray=True)
grey_img.shape  # (330, 220)
print(grey_img.min(), grey_img.max())  # 0 1 -> float!!
grey_img = skimage.img_as_ubyte(grey_img)
grey_img  # uint 8 -> 0 ~ 255
plt.imshow(grey_img)


# slice ----
print(grey_img[10:15, 20:25])  # Values from a slice

# overall summarise ----
mean_grey = grey_img.mean()
max_value = grey_img.max()
min_value = grey_img.min()
print(mean_grey, min_value, max_value)

# flipped ----
# 對 rgb or grey img 都 ok
flipped_img_LR = np.fliplr(grey_img)
flipped_img_UD = np.flipud(grey_img)

plt.subplot(2, 1, 1)
plt.imshow(grey_img, cmap="Greys")
plt.subplot(2, 2, 3)
plt.imshow(flipped_img_LR, cmap="Blues")
plt.subplot(2, 2, 4)
plt.imshow(flipped_img_UD, cmap="hsv")

# For all other options: https://matplotlib.org/tutorials/colors/colormaps.html

# Rotation ----
# 對 rgb or grey img 都 ok
rotated_img = ndimage.rotate(grey_img, 45)
# (330, 220) -> (389, 389)，他是對應到 pillow 旋轉後的 expand 方法
print(grey_img.shape, rotated_img.shape)
plt.imshow(rotated_img)

rotated_img_noreshape = ndimage.rotate(grey_img, 45, reshape=False)
# (330, 220) -> (330, 220)，他是對應到 pillow 旋轉後，不要 expand 的方法
print(grey_img.shape, rotated_img_noreshape.shape)
plt.imshow(rotated_img_noreshape)

# Filtering -----------
# 可以對 rgb or gray img 做，做之前都是 uint8, 做完之後也是 uint8
# Local filters: replace the value of pixels by a function of the values of neighboring pixels.

grey_img = skimage.io.imread("images/monkey.jpg", as_gray=True)
grey_img = skimage.img_as_ubyte(grey_img)

# Gaussian filter: from scipy.ndimage
# Gaussian filter smooths noise but also edges
gaussian_filtered = ndimage.gaussian_filter(grey_img, sigma=7)
plt.imshow(gaussian_filtered)
grey_img.dtype  # uint8
gaussian_filtered.dtype  # uint8

# Median filter is better than gaussian. A non-local means is even better
median_img = ndimage.median_filter(grey_img, 3)
plt.imshow(median_img)

# Sobel filter (edge detection)
img2 = skimage.img_as_ubyte(
    skimage.io.imread("images/test_images/aeroplane/1.jpg", as_gray=True))
plt.imshow(img2)
# Axis along which to calculate sobel
sobel_img_x = ndimage.sobel(img2, axis=0)
plt.imshow(sobel_img_x)
# Axis along which to calculate sobel
sobel_img_y = ndimage.sobel(img2, axis=1)
plt.imshow(sobel_img_y)

# for a list of all filters
# https://docs.scipy.org/doc/scipy/reference/ndimage.html

# backup -----
# img = img_as_ubyte(
#     io.imread("images/nucleiTubolin_small_noisy.jpg", as_gray=True))
# img1 = img_as_ubyte(io.imread("images/test_image.jpg", as_gray=True))
# img2 = skimage.img_as_ubyte(
#     skimage.io.imread("images/test_images/aeroplane/1.jpg", as_gray=False))

# uniform_filtered_img = ndimage.uniform_filter(img, size=9)
# # plt.imshow(uniform_filtered_img)

# # Gaussian filter: from scipy.ndimage
# # Gaussian filter smooths noise but also edges

# blurred_img = ndimage.gaussian_filter(img, sigma=3)  # also try 5, 7
# # plt.imshow(blurred_img)

# # Median filter is better than gaussian. A non-local means is even better
# median_img = ndimage.median_filter(img, 3)
# # plt.imshow(median_img)

# # Edge detection
# sobel_img = ndimage.sobel(img2, axis=0)  # Axis along which to calculate sobel
# # plt.imshow(sobel_img)
