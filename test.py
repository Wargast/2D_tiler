import time
import cv2
import numpy as np
import logging

from main import *
from seg_tools import *

logging.basicConfig(level=logging.INFO)

tex_tile = cv2.imread("datas/Textures/freeTexture3.png")
# tex_tile = cv2.cvtColor(tex_tile_color, cv2.COLOR_BGR2RGB)

w = 800
h = 1000
pt_list = []
cv2.namedWindow("test")
img = np.zeros((w,h,3), np.uint8)
tex_tile = cv2.resize(tex_tile, (200,200))
img = place_background(img, tex_tile)

segmented_img = segment(img)
rocks = get_connected_comp(segmented_img, threshold=30)
rock_matrix = np.zeros(segmented_img.shape)
# for rock in rocks:
#     i, j = rock.x + int(rock.w/2), rock.y + int(rock.h/2)
#     cv2.circle(img,(i,j),5,(0,255,0),cv2.FILLED)
#     if 0<i<rock_matrix.shape[0] and 0<j<rock_matrix.shape[1]:
#         rock_matrix[i,j] = 1
        
# cv2.imshow("test", segmented_img*255)
# cv2.imshow("test_loc", img)




cv2.waitKey()