import cv2

import numpy as np
from matplotlib import pyplot as plt

im_path = "D:\baboon.png"

im_array = cv2.imread(im_path)

plt.imshow(im_array)
plt.show()
red_threshold=200
red_mask=im_array[:,:,0]>red_threshold
im_array[np.where(red_mask)]=[0,0,0]
plt.imshow(im_array)
plt.show()