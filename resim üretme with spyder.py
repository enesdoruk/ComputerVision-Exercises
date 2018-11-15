
import cv2
import numpy as np
from matplotlib import pyplot as plt 






img = cv2.imread('1.jpg') 
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)







j=0
for batch in datagen.flow(dst,batch_size=1,
                          save_to_dir='resim4',
                        save_format='jpeg'):
    j+=1
    if j > 10:
        break

