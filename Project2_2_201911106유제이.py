import os,time
import numpy as np
from PIL import Image
from numpy import median
from matplotlib import pyplot as plt

def count_labels(reginfo):
    count = 0
    for i in range(len(reginfo)):
        if reginfo[i][0] != 0:
            count += 1
    return count

def intlimitimg(img2d):
    row,col=img2d.shape
    for i in range(row):
        for j in range(col):
            if img2d[i][j]>255:
                img2d[i][j]=255
            elif img2d[i][j]<0:
                img2d[i][j]=0
    return img2d            

def color3dtogray2d(cimg,gimg):
    global row, col
    for i in range(row):
        for j in range(col):
            gimg[i][j] = int(0.299*cimg[i][j][0]+0.587*cimg[i][j][1]+0.114*cimg[i][j][2])
    return gimg

def readimage(img_name):
    global row,col
    im = Image.open(img_name)
    col,row = im.size
    cimg = np.array(im)
    return cimg,row,col

def writeimage2d(img,name):
    row,col = img.shape
    intlimitimg(img)
    img=np.uint8(img)
    im=Image.fromarray(img)
    im.save(name)
    return im

# Morphology filter
def morphologyf(image,index):
    row, col = np.shape(image)
    buffer = np.full((row,col),0)

    str = ((0,0,1,1,1,0,0),
           (0,1,1,1,1,1,0),
           (0,1,1,1,1,1,0),
           (1,1,1,1,1,1,1),
           (0,1,1,1,1,1,0),
           (0,1,1,1,1,1,0),
           (0,0,1,1,1,0,0))
    
    str = np.array(str)
    str_size = 7
    str_area = 33
    imgbackground = 255

    for i in range(row-int(str_size)):
        for j in range(col-int(str_size)):
            count = 0
            for k in range(str_size):
                for l in range(str_size):
                    if str[k][l]!=0 and image[i+k][j+l]!=imgbackground:
                        count += 1
            
            # dilation
            if index == 0: 
                if count == str_area:
                    buffer[i+int(str_size/2)][j+int(str_size/2)] = 255 - imgbackground
                else:
                    buffer[i+int(str_size/2)][j+int(str_size/2)] = imgbackground
           # erosion
            elif index == 1:
                if count > 0:
                    buffer[i+int(str_size/2)][j+int(str_size/2)] = 255 - imgbackground
                else:
                    buffer[i+int(str_size/2)][j+int(str_size/2)] = imgbackground
    return buffer

# Region growing
def merge(i,j,ii,jj,pl):  # it does not perform merging between regions which should follow later.
    global rct, reg, reginfo

    reg[i][j][0]=pl

    if reg[ii][jj][0]==pl: 
        reginfo[int(pl)][0]+=1
        reginfo[int(pl)][1]=(reginfo[int(pl)][1]*(reginfo[int(pl)][0]-1)+reg[i][j][1])/reginfo[int(pl)][0]
    
    if reg[ii][jj][0]!=pl: # in case of second merging after merging once with upper or left pixel.
        old=reg[ii][jj][0]
        reg[ii][jj][0]=pl
        reginfo[int(pl)][0]+=1
        reginfo[int(pl)][1]=(reginfo[int(pl)][1]*(reginfo[int(pl)][0]-1)+reg[ii][jj][1])/reginfo[int(pl)][0]
        reginfo[int(old)][0]-=1
        
        if reginfo[int(old)][0]<=0:
            reginfo[int(old)][0]=0
            reginfo[int(old)][1]=0 
        else:
            reginfo[int(old)][1]=(reginfo[int(old)][1]*(reginfo[int(old)][0]+1)-reg[ii][jj][1])/reginfo[int(old)][0]
        
        mergeregion(old,pl)
    
    reg[i][j][1]=reginfo[int(reg[i][j][0])][1]
        
def mergeregion(l1, l2):
    global reg, reginfo, rct, row, col
    l1=int(l1)
    l2=int(l2)
    
    if l1 == l2 or reginfo[l1][1]-reginfo[l2][1] > th:
        print(f"Merging {l1} and {l2} failed.\n-Diff is",reginfo[l1][1]-reginfo[l2][1])
        return 0
    else:
        if l1 < l2:
            m,n=l1,l2
        else:
            m,n=l2,l1

        reginfo[m][0] += reginfo[n][0]
        reginfo[m][1] = (reginfo[m][1]*reginfo[m][0]+reginfo[n][1]*reginfo[n][0])/(reginfo[m][0]+reginfo[n][0])

        reginfo[n][0]=0
        reginfo[n][1]=0
        print(f"Merging {l1} and {l2} happened")

        for u in range(row):
            for v in range(col):
                if  reg[u][v][0]==n:
                        reg[u][v][0] = m
                        reg[u][v][1] = reginfo[m][1]
    
        return 1
    
def separate(i,j):
    global inimg, rct, reg, reginfo
    rct+=1
    reg[i][j][0]=rct
    reginfo[int(rct)][0]+=1
    reginfo[int(rct)][1]=inimg[i][j]
    

def relabeling():
    global reg,row, col, rct, reginfo
    ct=0
    reglabel = np.full((row*col,3),0.0)

    for i in range(rct+1):
        if reginfo[i][0]!=0:
            ct=ct+1
            reglabel[ct][0]=reginfo[i][0]
            reglabel[ct][1]=reginfo[i][1]
            reglabel[ct][2]=i # to save old label

    for i in range(rct+1):
            reginfo[i][0]=reglabel[i][0]
            reginfo[i][1]=reglabel[i][1]

    newreg=np.full((row,col,2),0.0)

    for i in range(row):
        for j in range(col):
            ol=reg[i][j][0]
            for k in range(rct+1):
                if reglabel[k][2]==ol:
                    newreg[i][j][0]=k
                    newreg[i][j][1]=reglabel[k][1]

    for i in range(row):
        for j in range(col):
                reg[i][j][0]=newreg[i][j][0]
                reg[i][j][1]=newreg[i][j][1]


def picklarge(img,info,n):
    row,col=img.shape
    
    regareasort=np.full(row*col,0)
    reglabelsort=np.full(n+1,0)
    outimg=np.full((row,col),255)

    for i in range(row*col):
        regareasort[i]=info[i][0]
    
    regareasort=np.sort(regareasort)[::-1]
    
    for i in range(n):
        for j in range(row*col):
            if regareasort[i]==info[j][0]:
                reglabelsort[i]=j

    for i in range(row):
        for j in range(col):
            for k in range(n):
                if img[i][j]==reglabelsort[k]:
                    outimg[i][j]=reglabelsort[k]
                    #outimg[i][j]=0
   
    return outimg

def reverse(image):
    image = image.convert("L")
    row, col = image.size
    img = np.array(image)
    x_center = int(row/2)
    y_center = int(col/2)
    count = 0
    rev = False

    for i in range(x_center - 3, x_center + 4):
        for j in range(y_center - 3, y_center + 4):
            if img[i][j] == 255:
                count += 1
    print(f"The white counted is : {count}")

    if count >= 25:  # the half size of 7*7 box
        # Get the maximum pixel intensity value
        max_intensity = 255

        # Invert the pixel intensity values
        inverted_image = Image.eval(image, lambda x: max_intensity - x)
        img = np.array(inverted_image)
        rev = True
    return img, rev

def reverse2(image):
    image = image.convert("L")
    row, col = image.size
    img = np.array(image)
    x_center = int(row/2)
    y_center = int(col/2)
    count = 0

    for i in range(x_center - 3, x_center + 4):
        for j in range(y_center - 3, y_center + 4):
            if img[i][j] == 255:
                count += 1
    print(f"The white counted is : {count}")

    if count < 25:  # the half size of 7*7 box
        # Get the maximum pixel intensity value
        max_intensity = 255

        # Invert the pixel intensity values
        inverted_image = Image.eval(image, lambda x: max_intensity - x)
        img = np.array(inverted_image)
    return img

def background_is_black(img):
    row, col = img.shape
    black = 0
    white = 0
    isBlack = False

    edge_x = [0, int(row/2), row-1]
    edge_y = [0, int(col/2), col-1]

    for i in range(len(edge_x)):
        for j in range(len(edge_y)):
            if img[i][j] == 255:
                white += 1
            else:
                black += 1
    
    if black > white:
        isBlack = True
    
    return isBlack

# main procedure starts here
os.system('cls' if os.name=='nt' else 'clear')
start = time.time()

# 1. read image
namelst = ["p11.png"]
for i in range(len(namelst)):
    img_name = namelst[i]
    cimg,row,col=readimage(img_name)

    # 2. convert into grayscale
    gimg = np.full((row,col),0)
    gimg = color3dtogray2d(cimg,gimg)

    # 3. binarization (Thresholding)
    inimg = gimg
    minv=min(map(min,gimg))
    maxv=max(map(max,gimg))
    thresh = int((maxv + minv)/2)
    print(f"min : {minv}, max : {maxv}, thresh : {thresh}")

    for i in range(row):
        for j in range(col):
            if inimg[i][j] > thresh:
                inimg[i][j] = 255
            else:
                inimg[i][j] = 0

    print("Input:\n",inimg)

    # 4. Morphology filter - opening
    inimg = morphologyf(inimg,1)
    inimg = morphologyf(inimg,0)

    # Resizing (To eliminate the edge line) 
    im = Image.fromarray(inimg)
    row, col = im.size
    im = im.crop((7,7,row-7,col-7))
    inimg = np.array(im)
    row, col = inimg.shape

    # 5. region growing, labeling, selection
    reg=np.full((row,col,2),0.0)  #reg[i][j][0] is label value of pixel (i,j), and reg[i][j][1] is updated intensity of pixel (i,j)
    reginfo = np.full((row*col,2),0.0) #reginfo[i][0] is the pixel count of region label i, and reginfo[i][1] is updated intensity of region label i
    reg[0][0][0]=1  # The label starts from 1 for first region
    rct = 0
    th = 0.5 # thresholding

    for i in range(row):
        for j in range(col):
            reg[i][j][1]=inimg[i][j]

    for i in range(row):
        for j in range(col):  # upper first and left next makes a better result as low label is prioritized.
            
            rowmerge=colmerge=0

            if i!=0 and abs(reg[i][j][1] - reginfo[int(reg[i-1][j][0])][1]) <= th:
                merge(i,j,i-1,j,reg[i-1][j][0])
                rowmerge=1

            if j!=0 and abs(reg[i][j][1] - reginfo[int(reg[i][j-1][0])][1]) <= th:
                if rowmerge!=1:
                    merge(i,j,i,j-1,reg[i][j-1][0]) # in case of col merge without prior row merge
                    colmerge=1

                else: # in case of col merge after prior row merge
                    if reg[i-1][j][0]!=reg[i][j-1][0]:  # to exclude duplicate merging for the same label upper and left pixels
                        if abs(reg[i][j][1] - reginfo[int(reg[i][j-1][0])][1]) <= th:
                            merge(i,j,i,j-1,reg[i-1][j][0])  # if rowmerge already happened, pl is set to previous rowmerge
                            colmerge=1

                if i!=0 and rowmerge==0 and abs(reginfo[int(reg[i][j][0])][1] - reginfo[int(reg[i-1][j][0])][1]) <= th*0.5: # After colmerge, if rowmerge becomes possible.
                    merge(i,j,i-1,j,reg[i][j-1][0]) # in case that colmerge happens, pl is set to previous colmerge
                    rowmerge=1

            if rowmerge==0 and colmerge==0:
                separate(i,j)

    relabeling()
    outregl=reg[:,:,0].copy().reshape(row,col)
    outregi=reg[:,:,1].copy().reshape(row,col)

    print("Region label:\n",outregl)
    print("Region intensity:\n",np.round(outregi,2))
    print("Region info:\n",np.round(reginfo[:rct+1,:],2))

    # 6. save image
    out_namei = "new3_ori_"+img_name
    writeimage2d(outregi,out_namei)