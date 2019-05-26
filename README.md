# Particle Image Velocimetry Demonstration
Simple Particle Image Velocimetry algorithm demo based on zero-mean cross correlation.

## Description 
### 1/ Images generation 
N gaussian particles located randomly in the first frame. Affine transformation applied to the first frame to generate the second one. Gaussian noise is added.  

### 2/ PIV 
Frames are subdivided in smaller region of interest (ROI). We compute the zero-mean cross correlation (ZMCC) with Fast Fourier Transform on each ROI two by two (ROI_frame_1, ROI_frame_2). The displacement is estimated by finding the location of the maximum in the result of the ZMCC.

## Example of motions
We tested different motions :
#### [1] Shearing 
#### [2] Translation 
#### [3] Rotation 

![alt amplitude spectrums](https://raw.githubusercontent.com/j-colas/piv_demo/master/result_piv.png)


