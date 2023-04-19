from skimage.color import rgb2gray
from skimage.io import imsave, imread
from skimage.filters import threshold_otsu
from skimage.transform import rotate
import os

image = imread("IsleOfMan.jpg")
image = rgb2gray(image)

thresh = threshold_otsu(image)
image = image > thresh

if not os.path.exists("images"):
    os.mkdir("images")

rotated_images = []
for i in range(12):
    imsave(f"images\\{i+1}.jpg",rotate(image, i*10))