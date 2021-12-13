import time
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import skimage
from skimage.data import coins
from skimage.transform import rescale

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.utils.fixes import parse_version
import logging

logging.basicConfig(level=logging.INFO)

if parse_version(skimage.__version__) >= parse_version("0.14"):
    rescale_params = {"anti_aliasing": False, "multichannel": False}
else:
    rescale_params = {}

tex_tile_color  = cv2.imread("datas/Textures/freeTexture3.png")
tex_tile = cv2.cvtColor(tex_tile_color, cv2.COLOR_BGR2GRAY)
# load the coins as a numpy array
# tex_tile = coins()

# Resize it to 20% of the original size to speed up the processing
# Applying a Gaussian filter for smoothing prior to down-scaling
# reduces aliasing artifacts.
smoothened_tile = gaussian_filter(tex_tile, sigma=2)
rescaled_tile = rescale(smoothened_tile, 0.2, mode="reflect", **rescale_params)

# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(rescaled_tile)

# Take a decreasing function of the gradient: an exponential
# The smaller beta is, the more independent the segmentation is of the
# actual image. For beta=1, the segmentation is close to a voronoi
beta = 10
eps = 1e-6
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

# Apply spectral clustering (this step goes much faster if you have pyamg
# installed)
N_REGIONS = 2 #39

# for assign_labels in ("kmeans", "discretize"):
t0 = time.time()
labels = spectral_clustering(
    graph, n_clusters=N_REGIONS, assign_labels="discretize", random_state=42
)
t1 = time.time()
labels = labels.reshape(rescaled_tile.shape)
title = "Spectral clustering: %.2fs", (t1 - t0)
print(title)
print(labels.shape)

i=0
j=0
for l in range(N_REGIONS):
    img = cv2.resize(tex_tile_color, labels.shape,interpolation=cv2.INTER_AREA)
    img[labels==l, :] = [255,0,0]
    cv2.namedWindow(f"res region {l}")
    cv2.moveWindow(f"res region {l}", i*300, j*100%800)
    cv2.imshow(f"res region {l}", img)
    i += 1
    if i*150 > 1000:
        i = 0
        j += 1
cv2.waitKey()