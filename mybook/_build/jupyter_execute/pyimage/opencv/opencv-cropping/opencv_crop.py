#!/usr/bin/env python
# coding: utf-8

# # Crop image

# In[4]:


import cv2
import matplotlib.pyplot as plt


# In[12]:


image = cv2.imread("adrian.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[16]:


plt.imshow(image);


# In[17]:


face = image[85:250, 85:220]


# In[18]:


plt.imshow(image);


# In[19]:


plt.imshow(face);


# In[20]:


body = image[90:450, 0:290]


# In[21]:


plt.imshow(body);


# * 如果是 bounding box，給的是 (xmin, ymin, xmax, ymax). 
# * 那 crop image 時，就用 `img[ymin:ymax, xmin:xmax]`
