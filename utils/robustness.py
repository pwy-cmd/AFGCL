import numpy
import sklearn
import cv2
import os
from skimage import io, data
import skimage

path = "/media/nuist/F64D5D03A2EB3420/Pwy/Code/Datasets/IMD/NIST/val"
save_path = "/media/nuist/F64D5D03A2EB3420/Pwy/Code/Datasets/IMD/NIST_noise/JPEG_100"
# re_gt = '/media/pwy/F64D5D03A2EB3420/Pwy/Code/Segmentation/data/COVER/valannot'
if not os.path.exists(save_path):
    os.makedirs(save_path)
for imgpath in os.listdir(path):
    image = cv2.imread(path + "/" + imgpath)
    # noise_img = skimage.util.random_noise(image, mode='gaussian', mean=0, var=0.5) * 255
    # noise_img = skimage.img_as_ubyte(noise_img)
    # cv2.imwrite(save_path + "/" + imgpath,noise_img)
    # noise_img = cv2.GaussianBlur(image,(3,3),0)
    # noise_img = cv2.medianBlur(image, 9)
    # noise_img = cv2.blur(image, (3, 3))
    cv2.imwrite(save_path + "/" + imgpath.split(".")[0] + ".jpg", image,[cv2.IMWRITE_JPEG_QUALITY,100])
    # noise_img = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    # cv2.imwrite(save_path + "/" + imgpath, noise_img)
