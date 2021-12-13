import cv2
import numpy as np
from scipy.special import comb
from seg_tools import Rock
from typing import List

def click_and_draw_point(event, x, y, flags, param):
    img = param[0]
    pt_list =param[1]
    window = param[2]
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x, y),5,(0,255,0),cv2.FILLED)
        cv2.imshow(window, img)
        pt_list.append((x,y))
	# check to see if the left mouse button was released

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    res_points = np.array([xvals,yvals], np.int32)

    return res_points.transpose()

def draw_curve(img,pts):
    # print("pts.shape", pts_interp.shape)
    # print(pts_interp[:10,:,:])
    pts = pts.reshape((-1,1,2))
    img = cv2.polylines(img, [pts], False, (255,0,0), 2)
    return img

def place_background(img, background_tex):
    w, h, c = img.shape
    w_tile, h_tile, c_tile = background_tex.shape
    for i in range(w//h_tile):
        for j in range( h//h_tile):
            # print((i,j))
            img[i*w_tile: (i+1)*w_tile, j*h_tile:(j+1)*h_tile,:] = background_tex
    return img

def place_texture_tiles(img, path_tex, background_tex, curve):
    w,h,c = img.shape
    w_texture, h_texture, c_texture = path_tex.shape
    first_tile = None
    grid = np.zeros((w//h_texture + 1, h//h_texture +1))
    print("grid size", grid.shape)

    for p in curve:
        i = p[1]//w_texture
        j = p[0]//h_texture
        print("p coord", p)
        print("grid coord", (i,j))
        grid[i,j] = 1

    print("fill grid")
    for i in range(grid.shape[0]-1):
        for j in range(grid.shape[1]-1):
            print((i,j))
            if grid[i,j]==1:
                img[i*w_texture: (i+1)*w_texture, j*h_texture:(j+1)*h_texture,:] = path_tex
    return img

def place_texture_rocks(img, rocks:List[Rock], rock_matrix, background_tex, curve, size):

    placement_mask = np.zeros(rock_matrix.shape)
    for p in curve:
        x = p[1]
        y = p[0]
        placement_mask[x-int(size/2):x+int(size/2), y-int(size/2):y+int(size/2)] = 1
    
    # cv2.imshow("placement_mask", placement_mask)
    rock_matrix = cv2.bitwise_and(placement_mask, rock_matrix)
    cv2.imshow("rock_matrix", rock_matrix)
    # cv2.waitKey()
    for r in rocks:
        x, y = r.x + int(r.w/2), r.y + int(r.h/2)
        if rock_matrix[y,x] == 1:
            print("print rock:", r.id)
            inv_mask = cv2.bitwise_not(r.mask)
            img = cv2.bitwise_and(img, img, mask=inv_mask)
            img += cv2.bitwise_and(background_tex, background_tex, mask=r.mask)
            
    return img
