import os
import cv2
import numpy as np
import torch.utils.data
import random
import albumentations as albu
from data.image_aug import image_aug


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes,
                 transform=None, transform_q=None, config=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.transform_q = transform_q
        self.config = config

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id))

        mask = []
        mask_id = img_id.split('.')[0] + self.mask_ext
        mask_img = cv2.imread(os.path.join(self.mask_dir, img_id.replace('jpg', 'png')), cv2.IMREAD_GRAYSCALE)

        mask.append(mask_img[..., None])
        mask = np.dstack(mask)

        img_q = image_aug(img, mask, self.config)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

            augmented_q = self.transform_q(image=img_q)
            img_q = augmented_q['image']
        # cv2.imshow('1', img)
        # cv2.imshow('2', img_q)
        # cv2.waitKey()
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        img_q = img_q.astype('float32') / 255
        img_q = img_q.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, img_q, mask, {'img_id': img_id}


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id))

        mask = []
        # mask_id = img_id.split('.')[0] + self.mask_ext
        mask_id = img_id
        # mask_img = cv2.imread(os.path.join(self.mask_dir, mask_id.replace('jpg', 'tif')), cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(os.path.join(self.mask_dir, mask_id.replace('jpg', 'png')), cv2.IMREAD_GRAYSCALE)
        mask.append(mask_img[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)

        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}
