import cv2
import numpy as np
from matplotlib import pyplot as plt

imgpath ='./img_examples/292.jpg'
img = cv2.imread(imgpath,3)
print (img.shape)

edges = cv2.Canny(img,100,200)
print (edges.shape)
plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('Original image'), plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap='gray')
plt.title('Edge Image'),plt.xticks([]),plt.yticks([])

plt.show()

