import time
import cv2
import numpy as np
import logging

from main import *
from seg_tools import *

logging.basicConfig(level=logging.INFO)

w = 800
h = 1000

tex_tile = cv2.imread("datas/Textures/freeTexture3.png")
tex_tile = cv2.resize(tex_tile, (200,200))
bg_tex = cv2.imread("datas/Textures/freeTexture2.png")
bg_tex = cv2.resize(bg_tex, (200,200))

# tex_tile = cv2.cvtColor(tex_tile_color, cv2.COLOR_BGR2RGB)
img = np.zeros((w,h,3), np.uint8)
img = place_background(img, bg_tex)

background_tex = np.zeros((w,h,3), np.uint8)
background_tex = place_background(background_tex, tex_tile)


segmented_img = segment(background_tex)
rocks, rocks_matrix = get_connected_comp(segmented_img, threshold=30)

curve = [(i,500) for i in range(100, 800)]
img = place_texture_rocks(img, rocks, rocks_matrix, background_tex, curve, size=100)

# for rock in rocks:
#     print(rock.id)
#     i, j = rock.x + int(rock.w/2), rock.y + int(rock.h/2)
#     cv2.circle(background_tex,(i,j),5,(0,255,0),cv2.FILLED)
#     # cv2.imshow("test", background_tex)

#     inv_mask = cv2.bitwise_not(rock.mask)
#     img = cv2.bitwise_and(img, img, mask=inv_mask)
#     img += cv2.bitwise_and(background_tex, background_tex, mask=rock.mask)
        
# cv2.imshow("test_mask", inv_mask)
cv2.imshow("test_loc", img)





cv2.waitKey()