import cv2
import numpy as np
from drawing_tools import *
from seg_tools import *

def main():
    w = 800
    h = 1000
    pt_list = []
    main_window = "main"
    cv2.namedWindow(main_window)
    cv2.moveWindow(main_window, 0, 0)
    
    # path_tex = cv2.imread("datas/Textures/freeTexture8.png")
    path_tex = cv2.imread("datas/Textures/freeTexture3.png")
    # path_tex = cv2.imread("datas/Textures/real_rocks.jpg")
    bg_tex = cv2.imread("datas/Textures/freeTexture2.png")
    
    path_tex = cv2.resize(path_tex, (150,150))
    # path_tex = cv2.blur(path_tex, (3,3))
    
    bg_tex = cv2.resize(bg_tex, (50,50))
    
    img = np.zeros((w,h,3), np.uint8)
    full_path_tex = np.zeros(img.shape, np.uint8)
    cv2.setMouseCallback(main_window, click_and_draw_point, (img, pt_list, main_window))
    
    img = place_background(img, bg_tex)
    full_path_tex = place_background(full_path_tex, path_tex)
    
    seg_full_path_tex = segment(full_path_tex)
    cv2.imshow("test", seg_full_path_tex*255)
    rocks, rocks_matrix = get_connected_comp(seg_full_path_tex, threshold=30, inv=True)
    
    cv2.imshow(main_window, img)

    while(1):
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('d') and len(pt_list)!=0 :
            img = place_background(img, bg_tex)
            pts = np.array(pt_list, np.int32)
            curve = bezier_curve(pts, nTimes=1000)
            img = place_texture_rocks(img, rocks, rocks_matrix, full_path_tex, curve, size=100)
            draw_curve(img, curve)
            cv2.imshow(main_window, img)
            pt_list.clear()


        if cv2.getWindowProperty(main_window,cv2.WND_PROP_VISIBLE) < 1:        
            break


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
    