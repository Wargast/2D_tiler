import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Rock:
    id: int
    mask: np.ndarray
    x: int
    y: int
    w: int
    h: int
    area: int
    centroid: Tuple[int, int]


def segment(img):
    pix_values = img.reshape((-1,3))
    pix_values = np.float32(pix_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    k = 2
    _, labels, (centers) = cv2.kmeans(pix_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)
    
    # convert all pixels to the color of the centroids
    mask = np.uint8(labels.reshape(img.shape[:2]))
        
    # segmented_image = centers[labels.flatten()].reshape(tex_tile.shape)
    
    return mask

def get_connected_comp(mask, threshold, verbose=False) -> List[Rock]:
    # apply connected component analysis to the thresholded image
    output = cv2.connectedComponentsWithStats(
        (mask), connectivity=8, ltype=cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    if numLabels < 10:
        mask = abs((mask-1))
        output = cv2.connectedComponentsWithStats(
            (mask), connectivity=8, ltype=cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
    
    rocks:List[Rock] = []
    rock_matrix = np.zeros(mask.shape)
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if verbose:
            if i == 0:
                text = "examining component {}/{} (background)".format(
                    i + 1, numLabels)
            # otherwise, we are examining an actual connected component
            else:
                text = "examining component {}/{}".format( i + 1, numLabels)
            print("[INFO] {}".format(text))
        
        if i!=0: 
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
                    
            componentMask = (labels == i).astype("uint8")*255
            # componentMask = cv2.cvtColor(componentMask,cv2.COLOR_GRAY2BGR)
            
            if area > threshold:
                rocks.append(Rock(i, componentMask, x, y, w, h, area, centroids[i]))
                x, y = x + int(w/2), y + int(h/2)
                if 0<y<rock_matrix.shape[0] and 0<x<rock_matrix.shape[1]:
                    rock_matrix[y, x] = 1
                
    print("tot rocks:", len(rocks))

    return rocks, rock_matrix