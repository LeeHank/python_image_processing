import numpy as np
from PIL import Image

# read
matplotlib_gray_img_uint8
matplotlib_color_img_uint8
pil_gray_img
pil_color_img


# convertion ---------------

# np.array to pil obj

# pil obj to np.array
pil_img_array = np.asarray(pil_img_obj)


# resize

# PIL method (Image object with .resize method)
img = Image.open("images/test_image.jpg")
print(img.size)
small_img = img.resize((200, 300))
# img_obj = img_array


# 18. image processing using pillow in python
