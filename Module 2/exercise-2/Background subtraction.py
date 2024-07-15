import numpy as np
import cv2
import matplotlib.pyplot as plt

greenbg = cv2.imread("GreenBackground.png")
background = cv2.imread("NewBackground.jpg")
obj = cv2.imread("Object.png")

obj = obj[:,:,[2,1,0]]
background = background[:,:,[2,1,0]]
greenbg = greenbg[:,:,[2,1,0]]

print( greenbg.shape )
print( background.shape )
print( obj.shape )

background = cv2.resize(background,(678,381))

mask = np.where(obj != greenbg ,255,0)

new_pic = np.where(mask==0,background,obj)
plt.imshow(new_pic)
plt.show()