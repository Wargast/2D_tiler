import time
import cv2
import numpy as np
import logging

from main import *
from seg_tools import *

logging.basicConfig(level=logging.INFO)

tex_tile_color  = cv2.imread("datas/Textures/freeTexture3.png")
tex_tile = cv2.cvtColor(tex_tile_color, cv2.COLOR_BGR2RGB)

pix_values = tex_tile.reshape((-1,3))
pix_values = np.float32(pix_values)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 2
_, labels, (centers) = cv2.kmeans(pix_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

print("labels.shape", labels.shape)
print("labels[:10]", labels[:10])
# convert back to 8 bit values
centers = np.uint8(centers)

# convert all pixels to the color of the centroids
segmented_masque = np.uint8(labels.reshape(tex_tile.shape[:2]))
# Invert labels
segmented_masque = abs((segmented_masque-1))
segmented_image = centers[labels.flatten()].reshape(tex_tile.shape)


get_connected_comp(segmented_masque, threshold=30)

# w = 800 
# h = 1000
# pt_list = []
# cv2.namedWindow("res_seg")
# cv2.imshow("tex originale", tex_tile_color)
# cv2.imshow("res_seg", segmented_image)
# cv2.imshow("res_seg_masque", segmented_masque)



# cv2.waitKey()