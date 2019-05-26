#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:10:19 2019

@author: jules
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage     import convolve
from scipy.ndimage     import affine_transform

#%%

def gen_particle(h):
    x = np.linspace(1,h,h)
    x0 = h/2
    w = h/8
    gauss = np.exp(-(x-x0)**2/(2*w**2))
    part  = np.zeros((h,h))
    for i in range(h):
        for j in range(h):
            part[i,j] = gauss[i]*gauss[j]
    return part

def piv(frame_1,frame_2,patch_size=64):
    [l,w] = patch_size, patch_size # patch size

    nx = int(L/l)
    ny = int(W/w)
    
    x0 = list()
    y0 = list()
    ux = list()
    uy = list()
    
    for i in range(nx):
        [imin,imax] = i*l,(i+1)*l
        if imax > L:
            imax = L
        for j in range(ny):
            [jmin,jmax] = j*l,(j+1)*l
            if jmax > L:
                jmax = L
                
            patch1 = frame_1[imin:imax,jmin:jmax]
            patch2 = frame_2[imin:imax,jmin:jmax]
            
            if np.std(patch1) != 0 :
                patch1 = (patch1-np.mean(patch1))/np.std(patch1)
            else : 
                patch1 = (patch1-np.mean(patch1))
            
            if np.std(patch2) != 0 :
                patch2 = (patch2-np.mean(patch2))/np.std(patch2)
            else : 
                patch2 = (patch2-np.mean(patch2))
           
            res = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(patch1)*np.conj(np.fft.fft2(patch2)))))
            pos_max = np.unravel_index(np.argmax(res, axis=None), res.shape)
            
            
            x0.append((imax+imin)/2)
            ux.append(pos_max[0]-l/2)
            y0.append((jmax+jmin)/2)
            uy.append(pos_max[1]-w/2)
            
    x0 = np.asarray(x0)
    y0 = np.asarray(y0)
    ux = np.asarray(ux)
    uy = np.asarray(uy)
    return [x0,y0,ux,uy]

#%%

[L,W] = 512,512
h     = 16 # condition => h%2 = 0

N = 2000 # number of particules

img = np.zeros((L,W))

part = gen_particle(h)

x_pos = np.random.randint(0,L,size=N)
y_pos = np.random.randint(0,W,size=N)
img[x_pos,y_pos] = 1

#%% TRANSFORMATION MATRIX 
### SHEAR
phi = -np.pi/70
Ms = np.array([[1,0,0],[np.tan(phi),1,0]])

### TRANSLATE
dx = 10
dy = -10
Mt = np.array([[1,0,dx],[0,1,dy]])

### ROTATE
theta = np.pi/70
Mr = np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0]])

#%% GENERATE FRAMES

frame_1 = convolve(img,part)
frame_2 = convolve(affine_transform(img,Ms),part)
frame_3 = convolve(affine_transform(img,Mt),part)
frame_4 = convolve(affine_transform(img,Mr),part)

std = .1
noise_1 = std*np.random.randn(*np.shape(frame_1))
noise_2 = std*np.random.randn(*np.shape(frame_2))
noise_3 = std*np.random.randn(*np.shape(frame_3))
noise_4 = std*np.random.randn(*np.shape(frame_4))

frame_1 += noise_1
frame_2 += noise_2
frame_3 += noise_3
frame_4 += noise_4

#%% PIV PROCESSING

[x01,y01,ux1,uy1] = piv(frame_1,frame_2)
[x02,y02,ux2,uy2] = piv(frame_1,frame_3)
[x03,y03,ux3,uy3] = piv(frame_1,frame_4)


#%% SHOW RESULTS

plt.close("all")
W = 5
L = 4
DPI = 150
plt.figure(num=1,figsize=(W,L),dpi=DPI,edgecolor='k',clear=True)
plt.set_cmap('Greys')
ax = [plt.subplot(3,3,i+1) for i in range(9)]

for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])


ax[0].imshow(frame_1)
ax[1].imshow(frame_2)
ax[2].imshow(np.zeros(img.shape))
ax[2].quiver(y01,x01,-uy1,ux1, pivot='mid', color='k')

ax[0].set_title("Frame 1")
ax[1].set_title("Frame 2")
ax[2].set_title("Displ. Field")

ax[3].imshow(frame_1)
ax[4].imshow(frame_3)
ax[5].imshow(np.zeros(img.shape))
ax[5].quiver(y02,x02,-uy2,ux2, pivot='mid', color='k')

ax[6].imshow(frame_1)
ax[7].imshow(frame_4)
ax[8].imshow(np.zeros(img.shape))
ax[8].quiver(y03,x03,-uy3,ux3, pivot='mid', color='k')


#plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig("result_piv.png")
