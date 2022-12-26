# https://www.youtube.com/watch?v=CTOURPZftuU

###########
# Let us start by looking at basic image transformation tasks like
# resize and rescale.
# Then let's look at a few ways to do edge detection.
# And then sharpening using deconvolution method and finally
# Then let's take a real life scenario like scratch assay analysis.


#Resize, rescale

from skimage import data, exposure
from skimage.feature import hog
from skimage.filters import threshold_otsu
from skimage.filters import try_all_threshold
from skimage.morphology import disk
from skimage.filters.rank import entropy
import numpy as np
import scipy.stats as st
from skimage import io, color, restoration, img_as_float
from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import io
import matplotlib.pyplot as plt

from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean

img = io.imread("images/test_image.jpg", as_gray=True)
img.shape  # (513, 639)
img.dtype  # float64


# Rescale, resize image by a given factor. While rescaling image
# gaussian smoothing can performed to avoid anti aliasing artifacts.
# Check rescales image size in variable explorer
img_rescaled = rescale(img, 1.0 / 4.0, anti_aliasing=False)
img_rescaled.shape  # (128, 160), 就是原本 (513,639) 的 1/4
plt.imshow(img_rescaled)


# Resize, resize image to given dimensions (shape)
img_resized = resize(img, (200, 200),  # Check dimensions in variable explorer
                     anti_aliasing=True)
img_resized.shape
plt.imshow(img_resized)


# Downscale, downsample using local mean of elements of each block defined by user
# downscale 的意思看下例就知
# 我如果把 rows (513) 變成原本的 1/4， columns (639) 變成原本的 1/3
# 那就等於將原本的圖，高的方向切成 513/4 = 129 條線, 寬的方向切成 639/3 = 213 條線，交織成 129*213個格子
# 那每個格子，都用該格內的像素的mean來代表，就可以做到 downsampling
# rows 變成原本的 1/4， columns 變成原本的 1/3，所以 rows 要切出
img_downscaled = downscale_local_mean(img, (4, 3))
plt.imshow(img_downscaled)
img_downscaled.shape  # (129, 213)

################################################


###############################
# Edge Detection


img = io.imread("images/test_image_cropped.jpg", as_gray=True)  # 0~1 float64
print(img.shape)
plt.imshow(img, cmap="gray")

edge_roberts = roberts(img)  # 0~1 float64
#plt.imshow(edge_roberts, cmap=plt.cm.gray, interpolation='nearest')
edge_sobel = sobel(img)  # 0~1 float64
edge_scharr = scharr(img)  # 0~1 float64
edge_prewitt = prewitt(img)  # 0~1 float64


fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                         figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(edge_roberts, cmap=plt.cm.gray)
ax[1].set_title('Roberts Edge Detection')

ax[2].imshow(edge_sobel, cmap=plt.cm.gray)
ax[2].set_title('Sobel')

ax[3].imshow(edge_scharr, cmap=plt.cm.gray)
ax[3].set_title('Scharr')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# Another edge filter is Canny. This is not just a single operation
# It does noise reduction, gradient calculation, and edge tracking among other things.
# Canny creates a binary file, true or false pixels.
edge_canny = feature.canny(img, sigma=3)
plt.imshow(edge_canny, cmap=plt.cm.gray)


###############################################
# Image deconvolution
# Uses deconvolution to sharpen images.
# 前提假設是，原本 blur 的影像，是光打上去時，有光暈 (不太確定...)


img = img_as_float(io.imread("images/BSE_Google_blurred.jpg"))
print(img.shape)  # (714, 901) gray image
plt.imshow(img)

# PSF = point spread function, 用來做 deconvolution

# point spread function to be used for deconvolution.
psf = np.ones((3, 3)) / 9
deconvolved, _ = restoration.unsupervised_wiener(img, psf)
plt.imshow(img, cmap='gray')
plt.imshow(deconvolved, cmap='gray')


# The following page was used as reference to generate the kernel
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm

def gkern(kernlen=21, nsig=2):  # Returns a 2D Gaussian kernel.

    lim = kernlen//2 + (kernlen % 2)/2
    x = np.linspace(-lim, lim, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


psf = gkern(5, 3)  # Kernel length and sigma
print(psf)


deconvolved, _ = restoration.unsupervised_wiener(img, psf)
plt.imshow(img, cmap='gray')
plt.imshow(deconvolved, cmap='gray')

#########################################
# Let's find a way to calculate the area of scratch in wound healing assay
# wound healing assay 是指 傷口癒合分析 (assay 是指化驗/分析)的意思

# Entropy filter
# e.g. scratch assay where you have rough region with cells and flat region of scratch.
# entropy filter can be used to separate these regions


img = io.imread("images/scratch.jpg")
print(img.dtype)  # uint8
print(img.shape)  # (289, 400)
plt.imshow(img, cmap="gray")


# Checkout this page for entropy and other examples
# https://scikit-image.org/docs/stable/auto_examples/

entropy_img = entropy(img, disk(3))
plt.imshow(entropy_img, cmap=plt.cm.gray)

# Once you have the entropy iamge you can apply a threshold to segment the image
# If you're not sure which threshold works fine, skimage has a way for you to check all

fig, ax = try_all_threshold(entropy_img, figsize=(10, 8), verbose=False)
plt.show()


# Now let us test Otsu segmentation.
# Just gives us a threshold value. Check in variable explorer.
thresh = threshold_otsu(entropy_img)
# let us generate a binary image by separating pixels below and above threshold value.
binary = entropy_img <= thresh
plt.imshow(binary, cmap=plt.cm.gray)
print("The percent white region is: ", (np.sum(binary == 1)*100)/(np.sum(binary ==
      0) + np.sum(binary == 1)))  # Print toal number of true (white) pixels

# We can do the same exercise on all images in the time series and plot the area to understand cell proliferation over time

###################################################################

# 沒講到這塊

# # HOG

# img = io.imread("images/Neuron.jpg", as_gray=False)
# print(img.shape)

# fd, hog_image = hog(img, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
#                     visualize=True, multichannel=True)
# print(hog_image.max())
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# ax1.axis('off')
# ax1.imshow(img, cmap=plt.cm.gray)
# ax1.set_title('Input image')

# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 50))

# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()
