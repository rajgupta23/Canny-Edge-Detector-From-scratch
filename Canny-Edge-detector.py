import skimage
import numpy as np
from matplotlib import pyplot as plt
# import skimage as sk 
from skimage.color import *
from skimage import io
from skimage import data
from skimage import filters
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage import metrics
from skimage.transform import rescale, resize, downscale_local_mean


def convu(img , ker):
  row, col = img.shape[0],img.shape[1]
  krow,kcol = ker.shape[0],ker.shape[1]
  kcx , kcy = kcol/2 , krow/2
  newimg = np.zeros((row,col))

  for i in range(0,row):
    for j in range(0,col):
      for m in range (0,krow):
        mm = (int)(krow - 1 - m)
        for n in range(0,kcol):
          nn = (int)(kcol-1-n)
          ii =(int)(i+kcy - mm)
          jj = (int)(j + kcx - nn)
          if(((ii>=0 and ii<row) and( jj>=0 and jj<col))):
            newimg[i][j] = newimg[i][j] + (img[ii,jj]* ker[mm][nn])
  
  # newimg = skimage.exposure.rescale_intensity(newimg, in_range=(0, 255))
  # ret = img_as_uint(newimg)

  return newimg


def gaussianfilter() : # 5x5 gauss filter.
    ret = np.ones((7,7))
    sig =1.0
    s = 2.0*sig*sig
    r = 0
    sum = 0.0
    for x in range(-2,5):
      for y in range(-2,5):
        r = np.sqrt(x*x + y*y)
        ret[x+2][y+2] = (np.exp(-(r*r)/s))/(np.pi*s)
        sum = sum + ret[x+2][y+2]
    return ret


def applysobel(gmat):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float64)
    dx = convu(gmat,kx)
    dy = convu(gmat,ky)
    gg = np.hypot(dx,dy)
    gg= (gg*255/gg.max())
    angtheta = np.arctan2(dy,dx)
    return (gg, angtheta)


def nonMaxSup(a,angle,rows,cols):
    angle *= (180 / np.pi)
    sup = np.zeros((rows,cols))
    angle[angle < 0] += 180
    for i in range(1,rows-1):
        for j in range(1,cols-1):
                if (0 <= angle[i][j] < 22.5) or (157.5 <= angle[i][j] <= 180):
                    max = a[i][j+1]
                    min = a[i][j+1]
                if (22.50 <= angle[i][j] < 67.50):
                    max = a[i+1][j-1]
                    min = a[i-1][j+1]
                if (67.50 <= angle[i][j] < 112.50):
                    max = a[i+1][j]
                    min = a[i-1][j]
                if (112.50 <= angle[i][j] < 157.50):
                    max = a[i-1][j-1]
                    min = a[i+1][j+1]

                if ((a[i][j] >= max) and (a[i][j] >= min)):
                    sup[i][j] = a[i][j]
                else:
                    sup[i][j] = 0
    return sup

def Thrashholcheck1(aa,ltr,htr,rows,cols):
  ht = aa.max() * htr
  lt = ht*ltr
  ret = np.zeros((rows,cols))
  for i in range(0,rows):
    for j in range(0,cols):
      if(aa[i][j]<lt):
        ret[i][j]=0
      elif(aa[i][j]>=ht):
        ret[i][j] = 255
      else:
        ret[i][j] = 25
  return ret



def hysteresis(img,weak,strong = 255):
  for i in range(1,img.shape[0]-1):
    for j in range(1,img.shape[1]-1):
      if(img[i][j]==255 or img[i][j]==0):
        continue
      if(img[i][j] == weak):
        if(img[i][j+1] ==255 or img[i+1][j] ==255 or img[i+1][j+1] ==255 or img[i-1][j-1] ==255 or img[i][j-1] ==255 or img[i-1][j] ==255 or img[i-1][j+1] ==255 or img[i+1][j-1] ==255):
          img[i][j] =255
        else:
          img[i][j] = 0
  return img;


def myCannyEdgeDetector(image, Low_Threshold, High_Threshold):
  image = rgb2grey(image)
  afterGuassian = convu(image,gaussianfilter())
  afterSobel,angle = applysobel(afterGuassian)
  afterNonMaxSup = nonMaxSup(afterSobel,angle,afterSobel.shape[0],afterSobel.shape[1])
  firstThrashold = Thrashholcheck1(afterNonMaxSup,Low_Threshold,High_Threshold,afterNonMaxSup.shape[0],afterNonMaxSup.shape[1])
  final = hysteresis(firstThrashold,25,255)
  fig = plt.figure(figsize=(10, 7))
  rows = 2
  columns = 2
  fig.add_subplot(rows, columns, 1)

  plt.imshow(image,cmap='gray')
  plt.axis('off')
  plt.title("Original")
  
  fig.add_subplot(rows, columns, 2)
  
  plt.imshow(final,cmap = 'gray')
  plt.axis('off')
  plt.title("After CED")
  # io.imshow(final,cmap = 'gray')

  image = ndi.rotate(image, 15, mode='constant')
  image = ndi.gaussian_filter(image, 4)
  image = random_noise(image, mode='speckle', mean=0.1)
  edges1 = feature.canny(image)

  img = resize(image, (final.shape[0], final.shape[1]),anti_aliasing=True)
  rag = -final.min()+ final.max()
  psnr = skimage.metrics.peak_signal_noise_ratio(img,final, data_range =  rag)
  print("PSNR = " ,end = '')
  print(psnr)
  print("SSIM = ",end='')
  print(skimage.metrics.structural_similarity(img, final,  data_range=rag))

  # return final


# read image 
im1 = io.imread('images/p1.jpeg')

# call funtions
myCannyEdgeDetector(im1,0.05,0.09)


