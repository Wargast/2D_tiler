import cv2
import numpy as np
from scipy.special import comb

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


    return res_points.transpose().reshape((-1,1,2))


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global img, pt_list
    # print("click ! ")
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x, y),5,(0,255,0),cv2.FILLED)
        cv2.imshow("test", img)
        pt_list.append((x,y))
	# check to see if the left mouse button was released
	
def draw_interpolation_curve(win,img,pts):
    pts = np.array(pt_list, np.int32)
    pts_interp = bezier_curve(pts, nTimes=1000)
    print("pts.shape", pts_interp.shape)
    print(pts_interp[:10,:,:])
    img = cv2.polylines(img, [pts_interp], False, (255,0,0), 2)
    cv2.imshow(win, img)



if __name__ == "__main__":
    w = 500
    h = 500
    pt_list = []
    cv2.namedWindow("test")
    cv2.setMouseCallback("test", click_and_crop)
    img = np.zeros((w,h,3), np.uint8)

    cv2.imshow("test", img)
    cv2.moveWindow("test", 10, 10)

    while(1):
        if cv2.waitKey(0) == ord('q'):
            break
        if cv2.waitKey(0) == ord('d'):
            draw_interpolation_curve("test", img, np.array(pt_list))
            pt_list.clear()

        if cv2.getWindowProperty('test',cv2.WND_PROP_VISIBLE) < 1:        
            break

    cv2.destroyAllWindows()