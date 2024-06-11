import os
import cv2
import random
import numpy as np


def find_vertex(mask):
    hor = np.where(np.sum(mask, axis=0) > 0)
    ver = np.where(np.sum(mask, axis=1) > 0)
    return hor[0][0], hor[0][-1], ver[0][0], ver[0][-1]


def image_aug(img, mask, config):
    img_size = img.shape[0] * img.shape[1]
    min_x, max_x, min_y, max_y = find_vertex(mask)

    tamper_region = img[min_y:max_y, min_x:max_x, :]
    tamper_size = tamper_region.shape[0] * tamper_region.shape[1]
    x_limit, y_limit = mask.shape[1], mask.shape[0]

    if config['img_q_gen'] == 'tamper_regions':
        pad_size = int(config['pad_size'])
        th = float(config['th_size'])
        while tamper_size / img_size < th:
            min_x = min_x - pad_size
            max_x = max_x + pad_size
            min_y = min_y - pad_size
            max_y = max_y + pad_size
            if min_x < 0:
                min_x = 0
            if min_y < 0:
                min_y = 0
            if max_x > x_limit:
                max_x = x_limit
            if max_y > y_limit:
                max_y = y_limit
            tamper_region = img[min_y:max_y, min_x:max_x, :]
            tamper_size = tamper_region.shape[0] * tamper_region.shape[1]
            tamper_region = img[min_y:max_y, min_x:max_x, :]
        return tamper_region

    elif config['img_q_gen'] == 'background_au':
        mask_temp = np.squeeze(mask)
        _, mask_temp = cv2.threshold(mask_temp, 127, 255, cv2.THRESH_BINARY)
        mask_temp = mask_temp // 255
        au_path = config['background_path']
        au_name = random.choices(os.listdir(au_path))[0]
        au_background = cv2.imread(os.path.join(au_path, au_name))
        au_background = cv2.resize(au_background, (img.shape[1], img.shape[0]))
        img_tg = img.copy()
        img_tg[:, :, 0] = img_tg[:, :, 0] * mask_temp
        img_tg[:, :, 1] = img_tg[:, :, 1] * mask_temp
        img_tg[:, :, 2] = img_tg[:, :, 2] * mask_temp
        mask_object = (mask_temp == 1)
        mask_object = mask_object[min_y:max_y, min_x:max_x]
        mask_3c = np.stack((mask_object, mask_object, mask_object), axis=2)
        au_background[min_y:max_y, min_x:max_x, :] = au_background[min_y:max_y, min_x:max_x, :] * (1 - mask_3c) + \
                                                     img[min_y:max_y, min_x:max_x, :] * mask_3c

        return au_background

    elif config['img_q_gen'] == 'orginal_img':
        return img
