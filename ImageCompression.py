import numpy as np
import matplotlib.pyplot as plt
import skimage
from matplotlib import image
# from scipy.io import loadmat
from scipy import misc
from skimage import io
from Kmeans import Kmeans

img = image.imread('image.png')
print(img.shape)

image1 = img.reshape(img.shape[0] * img.shape[1], 3)
a = Kmeans(image1, 256)
Centroids, Output = a.fit(20)
image2 = []
for i in range(img.shape[0] * img.shape[1]):
    image2.append(Centroids[Output[i]])
image_compressed = np.asarray(image2)
print(image_compressed)
image_compressed2 = image_compressed.reshape(img.shape[0], img.shape[1], 3)
print(image_compressed2)
i10 = skimage.img_as_ubyte(image_compressed2, force_copy=False)
plt.title(' image_compressed 256 colors')
plt.imshow(i10)
plt.show()
io.imsave('image_compressed_K256.png', i10)
# import scipy.misc
# misc.imsave('tiger_small.jpg', image_compressed)
