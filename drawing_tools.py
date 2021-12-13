import cv2
import numpy as np
from scipy.special import comb

def click_and_draw_point(event, x, y, flags, param):
    img = param[0]
    pt_list =param[1]
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x, y),5,(0,255,0),cv2.FILLED)
        cv2.imshow("test", img)
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
            print((i,j))
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
