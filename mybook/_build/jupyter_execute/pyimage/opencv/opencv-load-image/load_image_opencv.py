#!/usr/bin/env python
# coding: utf-8

# # Loading and Displaying Images

# ## 重點：

# ```python
# import cv2
# 
# # 讀檔
# img = cv2.imread("your_path_to_img.png")
# 
# # show image
# cv2.imshow("my title of image window", img) # 跳出 image window
# cv2.waitKey(0) # 要等鍵盤按下任意鍵，才會關閉視窗
# cv2.waitKey(3000) # 3000 毫秒後，關閉視窗
# 
# # write
# cv2.imwrite("path_to_save_file_name.png", img)
# ```

# ## py 檔

# * `load_image_opencv.py` 的內容

# ```python
# import argparse
# import cv2
# 
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())
# 
# # load the image from disk via "cv2.imread" and then grab the spatial
# # dimensions, including width, height, and number of channels
# image = cv2.imread(args["image"])
# (h, w, c) = image.shape[:3]
# 
# # display the image width, height, and number of channels to our
# # terminal
# print("width: {} pixels".format(w))
# print("height: {}  pixels".format(h))
# print("channels: {}".format(c))
# 
# # show the image and wait for a keypress
# cv2.imshow("Image", image)
# cv2.waitKey(0) # 按下任意 key 後，程式才會往下執行
# 
# # save the image back to disk (OpenCV handles converting image filetypes automatically)
# cv2.imwrite("newimage.jpg", image)
# ```

# * 在 terminal 中，執行： `python load_image_opencv.py --image jurassic_park.png`，就可以得到結果
