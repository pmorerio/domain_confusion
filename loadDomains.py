import numpy as np
import os
import skimage
from injectCircles import *
import random as rnd
import numpy.random as npr

datasets_dir = '/home/mzanotto/Desktop/theanoNets/Theano-Tutorials/data/'

def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h

def mnist(ntrain=60000,ntest=10000,onehot=True):
    data_dir = os.path.join(datasets_dir,'mnist/')
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = trX/255.
    teX = teX/255.

    X_normal = trX[:20000,:]
    X_circles = injectCircles(trX[20000:40000,:],noCircles=3)
    X_negative = 1 - trX[40000:,:]
       
    trD = np.hstack((np.zeros((20000),dtype='uint8'),np.ones((20000),dtype='uint8')))
    teD = 2*np.ones((20000),dtype='uint8')
    
    trD_fake = np.uint8(npr.rand(40000)*3)
    teD_fake = np.uint8(npr.rand(20000)*3)
                
    trX = np.vstack((X_normal,X_circles))
    teX = X_negative
    
    teY = trY[40000:]
    trY = trY[:40000]
        
    if onehot:
        trY = one_hot(trY, 10)
        trD = one_hot(trD,3)
        trD_fake = one_hot(trD_fake,3)
        teY = one_hot(teY, 10)
        teD = one_hot(teD,3)
        teD_fake = one_hot(teD_fake,3)

    else:
		trY = np.asarray(trY)
        
    
    return trX,teX,trY,teY,trD,teD,trD_fake,teD_fake  




