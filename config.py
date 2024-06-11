import argparse
from utils.tools import str2bool


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=7, type=int, metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--model_arch', default='MSVG_TRM', help='MSVG, MSVG_TRM')
    parser.add_argument('--encoder_name', default='resnext101_32x8d', type=str)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_w', default=512, type=int, help='image width')
    parser.add_argument('--input_h', default=512, type=int, help='image height')

    # CL model
    parser.add_argument('--down_sampling_factors', default=5, type=int, help='5,4,3,2')
    parser.add_argument('--num_ftrs', default=2048, help='output of the prediction and projection heads')
    parser.add_argument('--proj_hidden_dim', default=2048, help='hidden dim of the projection heads')
    parser.add_argument('--pred_hidden_dim', default=1024, help='hidden dim of the prediction heads')

    # dataset
    parser.add_argument('--train_dataset', default='Pretrain_60k', help='trian dataset name: CASIA, NIST, Coverage_aug')
    parser.add_argument('--test_dataset', default='NIST', help='test dataset name: CASIA, NIST, Coverage')
    parser.add_argument('--path', default='/home/panwy/Pwy/Dataset/IMD-DG', help='dataset path')
    parser.add_argument('--background_path', default='/home/panwy/Pwy/Dataset/CASIA/Au',
                        help='For ablantion study')
    parser.add_argument('--img_ext', default='.tif', help='image file extension')
    parser.add_argument('--mask_ext', default='.tif', help='mask file extension')
    parser.add_argument('--img_q_gen', default='tamper_regions',
                        help='The image_q generation method: tamper_regions, grayscale, orginal_img')
    parser.add_argument('--aug_method', default='all',
                        help='The data augmentation method: crop, filp, color, noise, all')
    parser.add_argument('--pad_size', default='20', help='work if the img_q_gen is not orginal_img')
    parser.add_argument('--th_size', default='0.6', help='work if the img_q_gen is not orginal_img')

    # finetune
    parser.add_argument('--pretrain_path', default='/home/panwy/Pwy/IMD/AF-CL-master/cpkt/MSVG_TRM/Pretrain/Pretrain_60k_IMD20_83.88_2024_05_18_21_08/model.pth')
    # parser.add_argument('--pretrain_path',
    #                     default='')

    # optimizer
    parser.add_argument('--optimizer', default='SGD', choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + '(default: Adam)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=12, type=int)

    config = parser.parse_args()

    return config
