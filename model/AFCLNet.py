from typing import Optional

import dgl
import networkx
import torch
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from segmentation_models_pytorch.base.heads import *
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from config import parse_args

config = vars(parse_args())


def HGRL(fm):
    node_number = fm.shape[-1] ** 2
    graph = networkx.complete_graph(node_number).to_directed()
    graph = dgl.from_networkx(graph).to('cuda:0')
    in_plant = fm.shape[1]
    GCN_Edge = GCN(in_feats=in_plant, hidden_size=in_plant // 2).to('cuda:0')
    gcn_feature = GCN_Edge(fm, graph)
    gcn_feature = gcn_feature + fm
    return gcn_feature


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super(GCN, self).__init__()
        self.GraphConv1 = GraphConv(in_feats, hidden_size)
        self.GraphConv2 = GraphConv(hidden_size, in_feats)

    def forward(self, inputs, g):
        b, c, h, w = inputs.shape
        inputs = inputs.view(b, c, -1)  # B x C x N
        # Expected dimension for GCN: N,*,hidden_size
        inputs = inputs.permute(2, 0, 1)  # N x B x C
        output = self.GraphConv1(g, inputs)
        output = torch.relu(output)
        output = self.GraphConv2(g, output)
        output = output.permute(1, 2, 0)
        output = output.view(b, c, h, w)
        return output


class SegmenHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


def Graph_Dec(planes):
    layer = torch.nn.Sequential(torch.nn.Conv2d(planes, planes // 8, kernel_size=1, bias=False),
                                torch.nn.BatchNorm2d(planes // 8),
                                torch.nn.Conv2d(planes // 8, planes // 16, kernel_size=1, bias=False),
                                torch.nn.BatchNorm2d(planes // 16),
                                torch.nn.Conv2d(planes // 16, 1, kernel_size=1, bias=False),
                                torch.nn.BatchNorm2d(1))
    return layer


def Graph_Downsample(planes, number):
    layer = []
    for i2 in range(1, number):
        layer.append(torch.nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1))
        layer.append(torch.nn.BatchNorm2d(planes))
    return torch.nn.Sequential(*layer)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Segmentation(SegmentationModel):
    def __init__(self):
        super().__init__()
        planes2 = 256
        planes3 = 512
        planes4 = 1024
        planes5 = 2048
        down_sample_factor = config['down_sampling_factors']
        # self.Graph_Down_p2 = Graph_Downsample(planes2, down_sample_factor)
        # self.Graph_Down_p3 = Graph_Downsample(planes3, down_sample_factor - 1)
        # self.Graph_Down_p4 = Graph_Downsample(planes4, down_sample_factor - 2)
        self.Graph_Down_p5 = Graph_Downsample(planes5, down_sample_factor - 3)

        # self.Graph_Dec_p2 = Graph_Dec(planes2)
        # self.Graph_Dec_p3 = Graph_Dec(planes3)
        # self.Graph_Dec_p4 = Graph_Dec(planes4)
        self.Graph_Dec_p5 = Graph_Dec(planes5)
        # self.Attention_p5 = ChannelAttention(planes5, ratio=8)
        self.con2d = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
        self.cat = torch.nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.segmentationhead = SegmenHead(in_channels=128, out_channels=1, kernel_size=1, upsampling=4)
        self.activation = torch.nn.Sigmoid()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        from lightly.models.modules import SimSiamProjectionHead, SimSiamPredictionHead
        num_ftrs = 2048
        proj_hidden_dim = 2048
        pred_hidden_dim = 1024
        out_dim = proj_hidden_dim
        self.projection_head = SimSiamProjectionHead(
            num_ftrs, proj_hidden_dim, out_dim
        )
        self.prediction_head = SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim
        )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # self.check_input_shape(x)

        features = self.encoder(x)
        down_p5 = self.Graph_Down_p5
        graph_p5 = down_p5(features[-1])
        graph_p5 = HGRL(graph_p5)
        trm = graph_p5
        dec_p5 = self.Graph_Dec_p5
        graph_p5 = dec_p5(graph_p5)
        graph_p5 = F.interpolate(graph_p5, size=x.size()[2:], mode='bilinear', align_corners=True)

        decoder_output = self.decoder(*features)
        masks = self.segmentationhead(decoder_output)
        trm_feat = self.avgpool(trm)
        feature_p5 = self.avgpool(features[-1])
        f = feature_p5.flatten(start_dim=1)
        z = self.projection_head(f)
        # get predictions
        p = self.prediction_head(z)
        # stop gradient
        z = z.detach()

        return masks, [graph_p5], z, p, trm_feat


class AFCL(Segmentation):
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

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "fpn-{}".format(encoder_name)
        self.initialize()
