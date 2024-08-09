import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
Car='BD/'
c = os.listdir(Car)
c.sort()
for ww in range(len(c)):
    if c[ww].count('ER')>0:
        Nomb=c[ww][:-6]
        print(Nomb)
        print(int((ww-1)/6)+1,'/',int((len(c)-1)/6))
        #Nomb='2024_01_16_15_30_29'
        image = cv2.imread('BD//'+Nomb+'ETK.png')
        X=np.shape(image)[0]
        Y=np.shape(image)[1]
        H=np.zeros((X,Y))
        L=np.load('BD//'+Nomb+'.npy')
        Sal=np.zeros((X,Y,8))
        for i in range(X):
            for j in range(Y):
                if image[i][j][0]==0 and image[i][j][1]==0 and image[i][j][2]==0:#Margen
                    H[i][j]=0
                    L[i,j,3]=0
                if image[i][j][0]==0 and image[i][j][1]==0 and image[i][j][2]==255:#azul
                    H[i][j]=1
                if image[i][j][0]==255 and image[i][j][1]==0 and image[i][j][2]==0:#rojo
                    H[i][j]=2
                if image[i][j][0]==0 and image[i][j][1]==255 and image[i][j][2]==0:#verde
                    H[i][j]=3
                if image[i][j][0]==128 and image[i][j][1]==128 and image[i][j][2]==0:#verde oscuro
                    H[i][j]=4
                if image[i][j][0]==0 and image[i][j][1]==255 and image[i][j][2]==255:#Turqueza
                    H[i][j]=5
                if image[i][j][0]==255 and image[i][j][1]==255 and image[i][j][2]==0:#amarillo
                    H[i][j]=6
                if image[i][j][0]==255 and image[i][j][1]==0 and image[i][j][2]==255:#rosado
                    H[i][j]=7
                if image[i][j][0]==128 and image[i][j][1]==128 and image[i][j][2]==128:#Fondo de la cinta
                    H[i][j]=8
                if image[i][j][0]==255 and image[i][j][1]==255 and image[i][j][2]==255:# Granos buenos
                    H[i][j]=9
                Sal[i,j,0]=L[i,j,0]
                Sal[i,j,1]=L[i,j,1]
                Sal[i,j,2]=L[i,j,2]
                Sal[i,j,3]=L[i,j,3]
                Sal[i,j,4]=image[i,j,0]
                Sal[i,j,5]=image[i,j,1]
                Sal[i,j,6]=image[i,j,2]
                Sal[i,j,7]=H[i,j]
                
        np.save(Nomb+'.npy', Sal)
'''
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(40,10))
ax1.imshow(np.dstack((Sal[:,:,2],Sal[:,:,1],Sal[:,:,0])).astype(np.uint8))
ax2.imshow(Sal[:,:,3].astype(np.uint8))
ax3.imshow(np.dstack((Sal[:,:,4],Sal[:,:,5],Sal[:,:,6])).astype(np.uint8))
ax4.imshow(Sal[:,:,7].astype(np.uint8))
plt.show()

mask = np.load('Segmentations\segm_1.npy')
img = np.load('MRIs\img_1.npy')
W=100
n=100
H=np.zeros((W,W))
for i in range(W):
    for j in range(W):
        H[i][j]=img[i+n][j+n]
        #print(img[i][j],';',end='')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,10))
ax1.imshow(H)
ax2.imshow(img)
ax3.imshow(mask)
plt.show()
'''
