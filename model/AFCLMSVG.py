from typing import Optional

import segmentation_models_pytorch
import torch
import torch.nn.functional as F
from lightly.models.modules import SimSiamProjectionHead, SimSiamPredictionHead

from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder

from segmentation_models_pytorch.base.heads import *
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from config import parse_args


class Seg_IDM(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_pyramid_channels: int = 256,
            decoder_segmentation_channels: int = 128,
            decoder_merge_policy: str = "add",
            decoder_dropout: float = 0.2,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None,

    ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "fpn-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):

        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # x = self.constrain_conv(x)
        features = self.encoder(x)

        p0 = features[0]  # [batch, 3, 320, 320]
        p1 = features[1]  # [batch, 64, 160, 160]
        p2 = features[2]  # [batch, 256, 80, 80]
        p3 = features[3]  # [batch, 512, 40, 40]
        p4 = features[4]  # [batch, 1024, 20, 20]
        p5 = features[5]  # [batch, 2048, 10, 10]

        enc_out = [p0, p1, p2, p3, p4, p5]

        feature_p2 = self.avgpool(p2)
        feature_p3 = self.avgpool(p3)
        feature_p4 = self.avgpool(p4)
        feature_p5 = self.avgpool(p5)
        down_enc_out = [feature_p2, feature_p3, feature_p4, feature_p5]

        decoder_output = self.decoder(*enc_out)
        feature_decoder = self.avgpool(decoder_output)
        masks = self.segmentation_head(decoder_output)

        return down_enc_out, masks, feature_decoder

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return


class SimSiam(torch.nn.Module):
    def __init__(
            self, backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(
            num_ftrs, proj_hidden_dim, out_dim
        )
        self.prediction_head = SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim
        )

        self.projection_head_g = SimSiamProjectionHead(
            num_ftrs, proj_hidden_dim, out_dim
        )
        self.prediction_head_g = SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim
        )

        self.projection_head_dec = SimSiamProjectionHead(
            128, 128, 128
        )
        self.prediction_head_dec = SimSiamPredictionHead(
            128, 64, 128
        )

    def forward(self, x):
        down_enc_out, masks, feature_decoder = self.backbone(x)
        # get projections
        f = down_enc_out[-1].flatten(start_dim=1)
        z = self.projection_head(f)
        # get predictions
        p = self.prediction_head(z)
        # stop gradient
        z = z.detach()

        return z, p, masks


def build_model(config):

    config = vars(parse_args())
    backbone = Seg_IDM(encoder_name=config['encoder_name'], encoder_weights="imagenet", activation=None,
                       in_channels=3, classes=1)
    num_ftrs = config['num_ftrs']
    out_dim = proj_hidden_dim = config['proj_hidden_dim']
    pred_hidden_dim = config['pred_hidden_dim']
    model = SimSiam(backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim)

    return model
