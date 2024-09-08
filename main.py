#!/bin/python

import sys
import cv2 as cv
import pyhist  # Histogram
import psnrssim  # PSNR + SSIM

HIST_THRESH = .99
PSNR_TRIGGER = 25
_SCAL_FACT = 1
SAVED_FRAMES = 0

def get_scal_down(videoHandle) -> float:
    width = int(videoHandle.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(videoHandle.get(cv.CAP_PROP_FRAME_HEIGHT))
    scal_down_width = 250/width
    scal_down_height = 250/height
    scal_down_fact = scal_down_height if scal_down_height >= scal_down_width  else scal_down_width
    return scal_down_fact

def preview_imgs(img1, img2):
    """Scaling down to fit in 250x250 """
    img1 = cv.resize(img1, None, fx=_SCAL_FACT, fy=_SCAL_FACT)
    img2 = cv.resize(img2, None, fx=_SCAL_FACT, fy=_SCAL_FACT)
    return cv.vconcat([img1, img2])

def save_image(img, number):
    global SAVED_FRAMES
    ret = cv.imwrite("frames/img_"+str(number)+".jpeg", img)
    if not ret:
        print("Error saving image. Directory 'frames' exists ?")
    SAVED_FRAMES = SAVED_FRAMES + 1

def main(filepath:str):
    frame_nb = 0
    video = cv.VideoCapture(filepath)
    global _SCAL_FACT
    _SCAL_FACT = get_scal_down(video)

    ret, img = video.read()
    save_image(img,frame_nb)
    prev_img = img

    while True:
        ret, img = video.read()
        frame_nb = frame_nb+1
        if ret is False:
            break

        hist_score = pyhist.diff(curr_img=img, prev_img=prev_img)
        if hist_score < HIST_THRESH:
            psnrssim_res = psnrssim.is_diff(curr_img=img, prev_img=prev_img)
            if psnrssim_res:
                #cv.imshow("Histogram Preview difference", preview_imgs(img, prev_img))
                #cv.waitKey(0)
                save_image(img,frame_nb)
                prev_img = img
    print(f"{SAVED_FRAMES} frames found.")
if __name__ == "__main__":
    main(sys.argv[1])