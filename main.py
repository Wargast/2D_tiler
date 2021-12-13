import cv2
import numpy as np
from drawing_tools import *

def main():
    w = 800
    h = 1000
    pt_list = []
    cv2.namedWindow("test")
    img = np.zeros((w,h,3), np.uint8)
    cv2.setMouseCallback("test", click_and_draw_point, (img, pt_list))
    path_tex = cv2.imread("datas/Textures/freeTexture3.png")
    bg_tex = cv2.imread("datas/Textures/freeTexture2.png")
    print(path_tex.shape)
    path_tex = cv2.resize(path_tex, (50,50))
    bg_tex = cv2.resize(bg_tex, (50,50))
    cv2.moveWindow("test", 0, 0)
    img = place_background(img, bg_tex)
    cv2.imshow("test", img)

    while(1):
        key = cv2.waitKey()
        if key == ord('q'):
            break
        if key == ord('d') and len(pt_list)!=0 :
            img = place_background(img, bg_tex)
            pts = np.array(pt_list, np.int32)
            pts_interp = bezier_curve(pts, nTimes=1000)
            img = place_texture_tiles(img, path_tex, bg_tex, pts_interp)
            draw_curve(img, pts_interp)
            cv2.imshow("test", img)
            pt_list.clear()


        if cv2.getWindowProperty('test',cv2.WND_PROP_VISIBLE) < 1:        
            break


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
    