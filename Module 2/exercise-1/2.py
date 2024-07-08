import matplotlib.image as mpimg
import numpy as np
img = mpimg.imread("download.jpg")
img = np.array(img)
gray_img_01 = ( np.max(img,axis=2) + np.min(img,axis=2) )/2
print( gray_img_01[0,0] )

gray_img_02 = ( np.mean(img,axis=2) )
print( gray_img_02[0,0])

gray_img_03 = 0.21*img[:,:,0] + 0.72*img[:,:,1] + 0.07*img[:,:,2]
print( gray_img_03[0,0])
