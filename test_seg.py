import time
import cv2
import numpy as np
import logging
from random import randint

from main import *
from seg_tools import *

logging.basicConfig(level=logging.INFO)

w = 600
h = 600

# path_tex = cv2.imread("datas/Textures/real_rocks.jpg")
path_tex = cv2.imread("datas/Textures/freeTexture3.png")
path_tex = cv2.blur(path_tex, (3,3))
tex_tile = cv2.resize(path_tex, (200,200))

background_tex = np.zeros((w,h,3), np.uint8)
background_tex = place_background(background_tex, tex_tile)

cv2.imshow(f"texture", tex_tile)

segmented_img = segment(background_tex)
rocks, rocks_matrix = get_connected_comp(segmented_img, threshold=30, inv=True)

cv2.imshow("seg", segmented_img*255)

img = np.zeros(background_tex.shape)
for rock in rocks:
    print(rock.id, rock.x, rock.y)
    i, j = rock.x + int(rock.w/2), rock.y + int(rock.h/2)
    # cv2.circle(background_tex,(i,j),5,(0,255,0),cv2.FILLED)
    # cv2.imshow("test", background_tex)

    rock_color = np.ones(background_tex.shape) * [randint(0, 255), randint(0, 255), randint(0, 255)]
    rock_color = np.uint8(rock_color)
    img += cv2.bitwise_and(rock_color, rock_color, mask=rock.mask)
    # cv2.imshow(f"rock {rock.id}", img2)
    
img = np.uint8(img)
cv2.imshow(f"rocks", img)



while True:
    key = cv2.waitKey(1)
    if cv2.getWindowProperty("seg",cv2.WND_PROP_VISIBLE) < 1:        
        break
    if key == ord('q'):
        break
cv2.destroyAllWindows()    
    

    