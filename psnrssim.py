#!/bin/python
 
import numpy as np
import cv2 as cv

PSNR_MIN_VALUE = 12
PSNR_MAX_VALUE = 25
SSIM_TRIGGER = 75

def getPSNR(I1, I2):
    s1 = cv.absdiff(I1, I2) #|I1 - I2|
    s1 = np.float32(s1)     # cannot make a square on 8 bits
    s1 = s1 * s1            # |I1 - I2|^2
    sse = s1.sum()          # sum elements per channel
    if sse <= 1e-10:        # sum channels
        return 0            # for small values return zero
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr

 
def getMSSISM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    # INITS
 
    I1 = np.float32(i1) # cannot calculate on one byte large values
    I2 = np.float32(i2)
 
    I2_2 = I2 * I2 # I2^2
    I1_2 = I1 * I1 # I1^2
    I1_I2 = I1 * I2
    # END INITS
 
    # PRELIMINARY COMPUTING
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)
 
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
 
    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
 
    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
 
    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
 
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
 
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
 
    ssim_map = cv.divide(t3, t1)    # ssim_map =  t3./t1;
 
    mssim = cv.mean(ssim_map)       # mssim = average of ssim map
    return mssim

def thresh_ssim(ssim, trigger):
    if ssim[2]*100 > trigger:  # R
        if ssim[1]*100 > trigger:  # G
            if ssim[0]*100 > trigger:  # B
                return False
    return True
 
 
def is_diff(curr_img, prev_img, psnr_trigger=PSNR_MAX_VALUE, ssim_trigger=SSIM_TRIGGER) -> bool:
    """ Compare two images using PSNR & MSSISM

        MSSISM is used when PSNR_MIN_VALUE < PSNR < trigger

        @return: True if image are different
    """
    psnrv = getPSNR(prev_img, curr_img)

    if psnrv < PSNR_MIN_VALUE:
        print(f"PSNR : {round(psnrv, 3)}dB")
        return True
    if psnrv > psnr_trigger:
        return False

    mssimv = getMSSISM(prev_img, curr_img)
    thresh_bool = thresh_ssim(mssimv, ssim_trigger)
    if thresh_bool:
        print(f"MSSISM: R {round(mssimv[2]*100,2)}% G {round(mssimv[1] * 100, 2)}% B {round(mssimv[0] * 100, 2)}%")
        return True
    return False