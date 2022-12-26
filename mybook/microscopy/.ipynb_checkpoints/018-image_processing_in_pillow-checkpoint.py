# https://www.youtube.com/watch?v=uDNqNv2N-pY&t=

###########################


# Reading images
import glob
from PIL import Image
import numpy as np

img = Image.open("images/test_image.jpg")  # PIL object, not a numpy array
img  # show image
img.show()  # show on external default viewer (e.g. new window)

# properties
print(img.format)  # JPEG
print(img.mode)  # RGB
print(img. size)  # (639, 513) -> (width, height)
# 注意，和 numpy 的 shape 相反，shape 是 (row, col) = (height, width)
np.asarray(img).shape  # (513, 639, 3)

# Resize images ---------
small_img = img.resize((200, 300))
small_img.size  # (200, 300)
np.asarray(small_img).shape  # (300, 200, 3)

big_img = img.resize((1200, 1200))
big_img.size  # (1200, 1200)
np.asarray(big_img).shape  # (1200, 1200, 3)


# thumbnail ----------
# resize() 直接轉到你要的大小，不管 aspect ratio
# 如果你用 thumbnail，那他會先將 width 轉成你指定的值，然後 height 用原本的 aspect ratio 算出來的 height
# thumbnail 只能縮小，如果你給他的 (width, height) 比原本的大，他就直接無視你
img.thumbnail((200, 200))  # 直接 assign 了
img.size  # (200, 161)
np.asarray(img).shape  # (161, 200, 3)

img.thumbnail((1200, 1200))  # (1200, 1200) 大於 現在的 (200,161)，直接無視
img.size  # (200, 161)


# Cropping images ----------
img = Image.open("images/test_image.jpg")
cropped_img = img.crop((0, 0, 200, 300))  # crops from (0,0) to (200,300)
# 注意，這邊的 (0, 0) 到 (200, 300)，都是 (width, height) 的表示法，所以和熟悉的 numpy shape (row, col) = (height, width) 相反
# 這會有點反直覺，所以我不太喜歡這樣用。要 crop，直接用 numpy array 的 slice 來 crop 就好
cropped_img.size  # (200, 300)
np.asarray(cropped_img).shape  # (300, 200, 3)


# 疊合兩張影像 ----------
# this involves copying original image and pasting a second image on it
img1 = Image.open("images/test_image.jpg")
print(img1.size)  # (639, 513)
img2 = Image.open("images/monkey.jpg")
print(img2.size)  # (220, 330)
img2.thumbnail((200, 200))  # Resize in case the image is very large.

img1_copy = img1.copy()  # Create a copy of the large image
# 把 img2 貼到 img1_copy 上，從 img1_copy 的 (50, 50) 開始貼
img1_copy.paste(img2, (50, 50))
img1_copy


# Rotating images ----------
img = Image.open("images/test_image.jpg")
img.size  # (639, 513)

img_90_rot = img.rotate(90)
img_90_rot  # 逆時針 90 轉 90 度後，貼到 原圖 size 的黑背景圖上
img_90_rot.size  # 旋轉後，本來 size 應該變成 (513, 639)，但因為是貼到 原圖 size 的黑背景上
# 所以 size 仍是 (639, 513)，然後，旋轉後的 height 本來是 639，就被 crop 成 513

# 加了 expand，那就是會貼到符合最大 size 的 黑背景上
img_90_rot_expand = img.rotate(90, expand=True)
img_90_rot_expand
img_90_rot_expand.size  # (513, 639)


img_45_rot = img.rotate(45)
img_45_rot  # 原理同 img_90_rot.size 所述
img_45_rot.size  # (639, 513)

img_45_rot_expand = img.rotate(45, expand=True)  # 原理同 img_90_rot_expand
img_45_rot_expand
img_45_rot_expand.size  # (815, 815)


# Flipping or transposing images ----------
img = Image.open("images/monkey.jpg")  # easy to see that the image is flipped

img_flipLR = img.transpose(Image.FLIP_LEFT_RIGHT)
img_flipLR


img_flipTB = img.transpose(Image.FLIP_TOP_BOTTOM)
img_flipTB

# Color transforms, convert images between L (greyscale), RGB and CMYK
img = Image.open("images/test_image.jpg")
np.asarray(img).shape  # (513, 639, 3)


grey_img = img.convert('L')  # L is for grey scale
grey_img
np.asarray(grey_img).shape  # (513, 639)


# Many other tasks can be performed. Here is full documentation.
# https://pillow.readthedocs.io/en/stable/reference/Image.html


# Here is a way to automate image processing for multiple images.


path = "images/test_images/aeroplane/*.*"
for file in glob.glob(path):
    print(file)  # just stop here to see all file names printed
    # now, we can read each file since we have the full path
    a = Image.open(file)

    rotated45 = a.rotate(45, expand=True)
    rotated45.save(file+"_rotated45.png", "PNG")
