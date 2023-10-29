import os
import cv2
import glob
import progressbar


for path in progressbar.progressbar(list(glob.glob("data/IAM_S/all_images/*/*.png"))):
    image = cv2.imread(path)
    size = (int(image.shape[1] * 64 / image.shape[0]), 64)
    image = cv2.resize(image, size)
    cv2.imwrite(path.replace('png', 'jpg'), image)

#for path in progressbar.progressbar(list(glob.glob("data/IAM_S/all_images/*/*.jpg"))):
#    image = cv2.imread(path)
#    size = (int(image.shape[1] * 64 / image.shape[0]), 64)
#    # image = cv2.resize(image, size)
#    # cv2.imwrite(path.replace('png', 'jpg'), image)

