import skimage
import numpy as np
import random as rnd
from skimage.draw import circle

#~ import matplotlib.pyplot as plt

# Specifically designed for MNIST, so reshape is 28x28.

def injectCircles(imgs,noCircles=3,rndSeed=231):
    
    rnd.seed(rndSeed)
    
    imgs = np.reshape(imgs,(len(imgs),28,28))
    
    for i in range(len(imgs)):
        
        for c in range(noCircles): 
            
            # rnd radius between 2 and 5:
            
            r = rnd.randint(2,5)
            
            # rnd center 
            
            xc,yc = rnd.randint(0,28-5),rnd.randint(0,28-5)
            
            rr,cc = circle(xc,yc,r)
            
            imgs[i,rr,cc] = 0.5
            
    #~ plt.imshow(imgs[0,:,:],cmap='gray')
    #~ plt.show()
    imgs = np.reshape(imgs,(len(imgs),28*28))
    return imgs
    
    
    
    
    

