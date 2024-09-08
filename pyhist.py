#!/bin/python

import cv2 as cv
#import numpy as np
#from matplotlib import pyplot as plt

HISTCOMP_METHOD = cv.HISTCMP_CORREL

def diff(curr_img, prev_img, use_greyscal= True, method= HISTCOMP_METHOD) -> float:
    """ Compare two histograms and return score"""
    if use_greyscal:
        current_hist = __get_hist_grey(curr_img)
        previous_hist = __get_hist_grey(prev_img)
    else:
        raise("Hist:diff Color mode not implemented yet")
    comp_value = __hist_compare(current_hist, previous_hist, method)
    return comp_value

def __get_hist_grey(img):
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([img_grey],[0],None,[256],[0,256])
    #plt.figure(0)
    #plt.plot(hist)
    #plt.show()
    return hist
   # red, gre, blu = cv.split(img)
   # histr = cv.calcHist([img],[0],None,[256],[0,256])
   # histg = cv.calcHist([img],[1],None,[256],[0,256])
   # histb = cv.calcHist([img],[2],None,[256],[0,256])

def __hist_compare(h1, h2, method) -> float:
    """ Return d(h1, h2) using method """
    compare_value = cv.compareHist(h1, h2, method)
    return compare_value
