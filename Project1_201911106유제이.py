# clock_noise1
# pip install numpy
# pip install pillow
# pip install matplotlib

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def intlimit(var):
    minv = 0
    maxv = 255
    if var > maxv:
        var = maxv
    elif var < minv:
        vat = minv
    return var 

def readimage(img_name):
    global row,col
    im = Image.open(img_name)
    cimg = np.array(im)
    col,row = im.size
    return cimg,row,col

def saveimage(cimg,name):
    global row,col
    im=Image.fromarray(cimg)
    #Display the image
    im.show()
    im.save(name)
    return im

# Histoequation
def histoeq(gimg2d):
    buff2d = np.full((row,col),0)
    histo = np.full(256,0)
    pdf=np.full(256,0.0)
    cdf=np.full(256,0.0)

    for i in range(row):
        for j in range(col):
            histo[gimg2d[i][j]] +=1

    for i in range(256):
        pdf[i] = histo[i]/(row*col)
        if i==0:
            cdf[i] = pdf[i]
        else:
            cdf[i] = pdf[i]
            for j in range(i):
                cdf[i] += pdf[j]
        
    for i in range(row):
        for j in range(col):
            buff2d[i][j]=round(255.0*cdf[gimg2d[i][j]])

    return buff2d

# Smoothing part
def medianfiltering(img2d,ms):
    buff2d=img2d.copy()
    row,col=img2d.shape
    hs=int(ms/2) # mask must be square with odd dimension.
    
    for i in range(hs,row-hs):
        for j in range(hs,col-hs):
            temp = []
            for p in range(-hs,hs+1):
                for q in range(-hs,hs+1):
                    temp.append(buff2d[i+p][j+q])
            temp.sort()
            img2d[i][j]=temp[int(ms*ms/2)]

# read image
img_name="violin_cvn3.jpg"
cimg,row,col=readimage(img_name)

# convert into the grey scale
gimg2d = np.full((row,col),0)
gimg2d[:,:] = 0.299*cimg[:,:,0]+0.587*cimg[:,:,1]+0.114*cimg[:,:,2]

# mask declare
std = 3.5
msize = 3
mask = np.ones((msize, msize))

hs=int(msize/2) # msize must be odd number
for i in range(-hs,hs+1):
    for j in range(-hs,hs+1):
        mask[i+hs][j+hs] = (1.5/hs)*(1.0/(2*3.1416*std))*np.exp(-1.0*(i*i+j*j)/(2*std*std))

gimg2d = histoeq(gimg2d)
medianfiltering(gimg2d,msize)

# save image
cimg[:,:,0] = cimg[:,:,1] = cimg[:,:,2] = gimg2d[:,:]
out_img_name="heq_smoothing"+f"{msize}_{std}_"+img_name
saveimage(cimg,out_img_name)