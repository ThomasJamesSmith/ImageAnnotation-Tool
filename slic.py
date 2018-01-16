 # import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import time
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
 
# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))
 
# loop over the number of segments
#for numSegments in (100, 200, 300,550):
numSegments = 550
# apply SLIC and extract (approximately) the supplied number
# of segments
start_time = time.time()
segments = slic(image, n_segments = numSegments, sigma = 5)
finish_time = time.time()
print("%d segments %d seconds" % (numSegments, finish_time - start_time))

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
mask = mark_boundaries(image, segments)
io.imsave("test_sp_mask.jpg", mask)
ax.imshow(mask)
#cv2.imwrite("test_sp_mask.jpg", mask)
plt.axis("off")
# show the plots
#plt.show()
#plt.savefig('test_sp_mask.jpg')