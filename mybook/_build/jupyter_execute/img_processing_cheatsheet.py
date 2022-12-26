#!/usr/bin/env python
# coding: utf-8

# # Image Processing Cheatsheet

# ## Read, save, show

# | package  | read function                                                                                                                                                      | show function                                                                                                  | save function                                                                                  | type                       | channel | dimension (color img)    | dimension (gray img)                                      | value range | value type |
# |----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|----------------------------|---------|--------------------------|-----------------------------------------------------------|-------------|------------|
# | plt      | import matplotlib.pyplot as plt<br><br>img_array = plt.imread("xx.jpg")                                                                                            | plt.show(img_array)                                                                                            | plt.imsave('xx.png', img_array)                                                                | ndarray                    | R, G, B | (height, width, channel) | (height, width)                                           | [0-255]     | uint8      |
# | PIL      | import np<br>from PIL import Image<br><br>img_obj = Image.open("xx.jpg")<br>img_array = np.asarray(img_obj)                                                        | img<br><br>plt.imshow(img_array)                                                                               | img_obj.save("xx.jpeg")<br><br>img_obj = Image.fromarray(img_array)<br>img_obj.save("xx.jpeg") | <br><br><br>PIL<br>ndarray | R, G, B | (height, width, channel) | (height, width, channel)<br># channel = 3<br># same value | [0-255]     | uint8      |
# | tf.keras | from tensorflow.keras.preprocessing.image import <br>load_img, img_to_array, array_to_img<br><br>img_obj = load_img("xx.jpg")<br>img_array = img_to_array(img_obj) | img<br>plt.imshow(img_array)                                                                                   | img_obj.save("xx.jpeg")<br><br>img_obj = array_to_img(img_array)<br>img_obj.save("xx.jpeg")    | <br><br>PIL<br>ndarray     | R, G, B | (height, width, channel) | (height, width, channel)                                  | [0.-255.]   | float32    |
# | cv2      | import cv2<br>img_array = cv2.imread("xx.jpg")                                                                                                                     | plt.imshow(img_array)<br><br>cv2.imshow("window_name", img_array)<br>cv2.waitKey(0)<br>cv2.destroyAllWindows() | cv2.imwrite('xx.png', img_array)                                                               | ndarray                    | B, G, R | (height, width, channel) | (height, width, channel)                                  | [0-255]     | uint8      |

# In[ ]:




