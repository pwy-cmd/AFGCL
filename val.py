import os
import argparse

import cv2
import yaml
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import albumentations as albu

from model.AFCLNet import AFCL
from model.AFCLMSVG import build_model
from utils.transforms import transforms
from data.base_dataset import ValDataset
from sklearn.metrics import roc_auc_score
from utils.tools import AverageMeter, str2bool
from utils.metrics import iou_score, calculate_pixel_f1, iou


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_dataset', default='CASIA1.0', help='test dataset')
    parser.add_argument('--save', default=True, help='save the output')
    parser.add_argument('--path', default="/home/panwy/Pwy/Dataset/IMD-DG", help='dataset path')
    parser.add_argument('--model_path', default='/home/panwy/Pwy/IMD/AFGCL-github/cpkt/Pretrain_60k_CASIA_58.2%_89.7%_2024_05_05_20_29', help='model name')
    args = parser.parse_args()

    return args


F1_score = []
AUC_score = []
MCC_score = []


def main():
    config = vars(parse_args())

    with open(os.path.join(config['model_path'], 'config.yml'), 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in model_config.keys():
        print('%s: %s' % (key, str(model_config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    if model_config['model_arch'] == 'MSVG_TRM':
        model = AFCL(encoder_name=model_config['encoder_name'], encoder_weights="imagenet",
                     in_channels=model_config['input_channels'], activation=None, classes=model_config['num_classes'])
    else:
        model = build_model(config)
    model = model.cuda()

    model.load_state_dict(torch.load(os.path.join(config['model_path'], 'model.pth')))
    model.eval()

    val_path = os.path.join(config['path'], config['test_dataset'], 'val')
    val_img_ids = os.listdir(val_path)
    valannot_path = os.path.join(config['path'], config['test_dataset'], 'valannot')

    val_transform = albu.Compose([albu.Resize(model_config['input_h'], model_config['input_w']), albu.Normalize()])

    val_dataset = ValDataset(img_ids=val_img_ids, img_dir=val_path, mask_dir=valannot_path,
                             img_ext=model_config['img_ext'], mask_ext='.png',
                             num_classes=model_config['num_classes'], transform=val_transform)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=0, drop_last=False)

    save_path = os.path.join('outputs', config['model_path'].split('/')[-1])
    os.makedirs(os.path.join(save_path), exist_ok=True)
    avg_meter = AverageMeter()
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            if model_config['model_arch'] == 'MSVG_TRM':
                output, output_g, z0, p0, trm_feats = model(input)

            if model_config['model_arch'] == 'MSVG':
                z, p, output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            # out_auc = output.squeeze().cpu().detach().numpy()
            out_auc = torch.sigmoid(output).squeeze().cpu().detach().numpy()

            output = torch.sigmoid(output).cpu().numpy()
            # output_g = torch.sigmoid(output_g[0]).cpu().numpy()
            pr_mask = output.squeeze().round()
            # pr_g_mask = output_g.squeeze().round()
            gt_mask = target.squeeze().cpu().detach().numpy()
            f1, p, r, mcc = calculate_pixel_f1(pr_mask.flatten(), gt_mask.flatten())
            # mIoU = iou_score(output, gt_mask)
            try:
                auc = roc_auc_score(gt_mask.astype('int').ravel(), out_auc.ravel())

            except ValueError:
                pass
            AUC_score.append(auc)
            F1_score.append(f1)
            MCC_score.append(mcc)
            if config['save']:
                # print(meta['img_id'][0])
                cv2.imwrite(os.path.join(save_path, meta['img_id'][0]), (pr_mask * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)
    F1 = np.mean(np.array(F1_score))
    AUC = np.mean(np.array(AUC_score))
    MCC = np.mean(np.array(MCC_score))
    print('|F1 score %5f |' % F1)
    print('|AUC score %5f |' % AUC)
    print('|MCC score %5f |' % MCC)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
