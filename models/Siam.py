import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn.quantized import BatchNorm2d
from models.FcaNet import fcanet34, fcanet50, fcanet101, fcanet152
from models.FPAN import FPAN, LastLevelP6P7
from models.FPN import FPN, LastLevelP6P7
from models.MultiHeads import CLSHead, REGHead, LDMHead
from models.Anchors import Anchors
from models.MultiLoss import MultiLoss
from utils.nms_wrapper import nms
from utils.box_coder import BoxCoder
from utils.bbox import clip_boxes, clip_landmarks
import torch.nn.functional as F
from models.CSModel import CSModel, LayerNorm2d
from models.Dense import CrossAttention_DenseAVInteractions
from models import FFTrans


def hook_fn(grad):
    with open("grad.txt", "a") as f:
        f.write(f"{grad.norm().item()}\t")


class Siam(nn.Module):
    def __init__(self, backbone='fca101', hyps=None):
        super(Siam, self).__init__()
        self.num_classes = 2
        self.anchor_generator = Anchors(
            ratios=np.array([0.2, 0.5, 1.0, 2.0, 5.0]),
        )
        self.num_anchors = self.anchor_generator.num_anchors
        self.init_backbone(backbone)
        self.inner_blocks_1 = nn.ModuleList()
        self.layer_blocks_1 = nn.ModuleList()
        self.inner_blocks_2 = nn.ModuleList()
        self.layer_blocks_2 = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.cs_models1 = nn.ModuleList()
        self.cs_models2 = nn.ModuleList()
        self.up_layers1 = nn.ModuleList()
        self.up_layers2 = nn.ModuleList()
        self.register_buffer("f_fuse", None)
        self.fusion_block = nn.ModuleList([])
        self.TFMamba = nn.ModuleList([])
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.norm3 = nn.ModuleList()
        self.norm4 = nn.ModuleList()
        self.inner_blocks_3 = nn.ModuleList()
        self.inner_blocks_4 = nn.ModuleList()

        stages = [['layer2'], ['layer3'], ['layer4']]
        stage_channels = [512, 1024, 2048]
        stage_scales = [96, 48, 24]
        # stage_scales = [128, 64, 32]
        for index, in_channels in enumerate(stage_channels):
            self.inner_blocks_1.append(nn.Conv2d(in_channels, 256, 1))
            self.inner_blocks_2.append(nn.Conv2d(in_channels, 256, 1))
            self.layer_blocks_1.append(nn.Conv2d(256, 256, 3, 1, 1))
            self.layer_blocks_2.append(nn.Conv2d(256, 256, 3, 1, 1))
            self.conv_layers.append(nn.Conv2d(in_channels // (2 ** (index + 1)), 256, 3, 2, 1))
            self.cs_models1.append(CSModel(in_channels=256))
            self.cs_models2.append(CSModel(in_channels=256))
            self.fusion_block.append(CrossAttention_DenseAVInteractions(stage_scales[index], 512, stages[index]))
            self.up_layers1.append(nn.Conv2d(256, in_channels, 1))
            self.up_layers2.append(nn.Conv2d(256, in_channels, 1))
            self.norm1.append(LayerNorm2d(normalized_shape=256))
            self.norm2.append(LayerNorm2d(normalized_shape=256))
            self.norm3.append(nn.BatchNorm2d(num_features=256))
            self.norm4.append(nn.BatchNorm2d(num_features=256))
            self.inner_blocks_3.append(nn.Conv2d(256, in_channels, 1))
            self.inner_blocks_4.append(nn.Conv2d(3, in_channels, kernel_size=1, stride=768//stage_scales[index], padding=0))


        self.fpan = FPN(
            in_channels_list=self.fpn_in_channels,
            out_channels=256,
            top_blocks=LastLevelP6P7(self.fpn_in_channels[-1], 256)
        )

        self.cls_head = CLSHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            num_anchors=self.num_anchors,
            num_classes=self.num_classes
        )

        self.reg_head = REGHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            num_anchors=self.num_anchors,
            num_regress=5
        )

        self.ldm_head0 = LDMHead(in_channels=256, feat_channels=256, num_anchors=self.num_anchors, num_landmarks=4,
                                 level='p3')
        self.ldm_head1 = LDMHead(in_channels=256, feat_channels=256, num_anchors=self.num_anchors, num_landmarks=4,
                                 level='p4')
        self.ldm_head2 = LDMHead(in_channels=256, feat_channels=256, num_anchors=self.num_anchors, num_landmarks=4,
                                 level='p5')
        self.ldm_head3 = LDMHead(in_channels=256, feat_channels=256, num_anchors=self.num_anchors, num_landmarks=4,
                                 level='p6')
        self.ldm_head4 = LDMHead(in_channels=256, feat_channels=256, num_anchors=self.num_anchors, num_landmarks=4,
                                 level='p7')
        self.loss = MultiLoss(func='lmr5p')
        self.box_coder = BoxCoder()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def init_backbone(self, backbone):
        if backbone == 'fca34':
            self.backbone1 = fcanet34(pretrained=True)
            self.backbone2 = fcanet34(pretrained=True)
            self.fpn_in_channels = [128, 256, 512]
            del self.backbone1.avgpool
            del self.backbone2.avgpool
            del self.backbone1.fc
            del self.backbone2.fc
        elif backbone == 'fca50':
            self.backbone1 = fcanet50(pretrained=True)
            self.backbone2 = fcanet50(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
            del self.backbone1.avgpool
            del self.backbone2.avgpool
            del self.backbone1.fc
            del self.backbone2.fc
        elif backbone == 'fca101':
            self.backbone1 = fcanet101(pretrained=True)
            self.backbone2 = fcanet101(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
            del self.backbone1.avgpool
            del self.backbone2.avgpool
            del self.backbone1.fc
            del self.backbone2.fc
        elif backbone == 'fca152':
            self.backbone1 = fcanet152(pretrained=True)
            self.backbone2 = fcanet152(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
            del self.backbone1.avgpool
            del self.backbone2.avgpool
            del self.backbone1.fc
            del self.backbone2.fc
        else:
            raise NotImplementedError


    def process_stages(self, c1, c2, stages, num):
        current_1, current_2 = c1, c2

        for idx, stage in enumerate(stages):
            backbone_layer1 = getattr(self.backbone1, stage)
            backbone_layer2 = getattr(self.backbone2, stage)

            next_1 = backbone_layer1(current_1)
            next_2 = backbone_layer2(current_2)

            next_1 = self.inner_blocks_1[num](next_1)
            next_2 = self.inner_blocks_2[num](next_2)
            next_1 = self.layer_blocks_1[num](next_1)
            next_2 = self.layer_blocks_2[num](next_2)

            if stages == ['layer2']:
                self.f_fuse = self.conv_layers[num](current_1) + self.conv_layers[num](current_2)
                self.f_fuse = self.norm1[num](self.f_fuse)
            else:
                self.f_fuse = self.conv_layers[num](self.f_fuse)

            f_fuse_tmp = self.fusion_block[num](next_1, next_2)
            f_fuse_tmp = self.norm2[num](f_fuse_tmp)


            next_1 = self.cs_models1[num](self.f_fuse, next_1)
            next_2 = self.cs_models2[num](self.f_fuse, next_2)
            next_1 = self.norm3[num](next_1)
            next_2 = self.norm4[num](next_2)

            if stages == ['layer2']:
                self.f_fuse = f_fuse_tmp
            else:
                self.f_fuse = f_fuse_tmp + self.f_fuse
                self.f_fuse = self.norm1[num](self.f_fuse)

            fusion_feature = self.inner_blocks_3[num](self.f_fuse)

            current_1, current_2 = next_1, next_2
            current_1 = self.up_layers1[num](current_1)
            current_2 = self.up_layers2[num](current_2)
        return current_1, current_2, fusion_feature


    def siamfft(self, ims_sar, ims_fft):
        ims_fft_magnitude = torch.abs(ims_fft)
        ims_fft_phase = torch.angle(ims_fft)

        c1_1 = self.backbone1.relu(self.backbone1.bn1(self.backbone1.conv1(ims_sar)))
        c1_2 = self.backbone2.relu(self.backbone2.bn1(self.backbone2.conv1(ims_fft_magnitude)))
        c2_1 = self.backbone1.layer1(self.backbone1.maxpool(c1_1))
        c2_2 = self.backbone2.layer1(self.backbone2.maxpool(c1_2))
        c3_1, c3_2, fusion_feature3 = self.process_stages(c2_1, c2_2, ['layer2'], num=0)
        c4_1, c4_2, fusion_feature4 = self.process_stages(c3_1, c3_2, ['layer3'], num=1)
        c5_1, c5_2, fusion_feature5 = self.process_stages(c4_1, c4_2, ['layer4'], num=2)

        ims_fft_phase_3 = self.inner_blocks_4[0](ims_fft_phase)
        ims_fft_phase_4 = self.inner_blocks_4[1](ims_fft_phase)
        ims_fft_phase_5 = self.inner_blocks_4[2](ims_fft_phase)

        c3_2 = torch.fft.ifft2(torch.fft.ifftshift(torch.polar(c3_2, ims_fft_phase_3)), norm="ortho").real
        c4_2 = torch.fft.ifft2(torch.fft.ifftshift(torch.polar(c4_2, ims_fft_phase_4)), norm="ortho").real
        c5_2 = torch.fft.ifft2(torch.fft.ifftshift(torch.polar(c5_2, ims_fft_phase_5)), norm="ortho").real

        c3 = c3_1 + c3_2
        c4 = c4_1 + c4_2
        c5 = c5_1 + c5_2

        return [c3, c4, c5]

    def forward(self, imgs, gt_boxes=None, gt_landmarks=None, test_conf=None, process=None):
        anchors_list, offsets_list, cls_list, var_list = [], [], [], []
        original_anchors = self.anchor_generator(imgs)
        anchors_list.append(original_anchors)
        imgs_fft = torch.fft.fftshift(torch.fft.fft2(imgs, norm='ortho'), dim=[-2, -1])

        feature = self.siamfft(imgs, imgs_fft)
        features = self.fpan(feature)

        cls_score = torch.cat([self.cls_head(feature) for feature in features], dim=1)
        bbox_pred = torch.cat([self.reg_head(feature) for feature in features], dim=1)
        land_pred = torch.cat([self.ldm_head0(features[0]), self.ldm_head1(features[1]), self.ldm_head2(features[2]),
                               self.ldm_head3(features[3]), self.ldm_head4(features[4])], dim=1)
        if self.training:
            losses = dict()
            losses['loss_cls'], losses['loss_reg1'], losses['loss_reg2'] = self.loss(cls_score, bbox_pred,
                                                                                     anchors_list[-1], gt_boxes,
                                                                                     landmarks=land_pred,
                                                                                     gt_linestrips=gt_landmarks,
                                                                                     iou_thres=0.5)
            return losses
        else:
            return self.decoder(imgs, anchors_list[-1], cls_score, bbox_pred, land_pred, test_conf=test_conf)



    def decoder(self, imgs, anchors, cls_score, bbox_pred, landmark_pred, thresh=0.6, nms_thresh=0.2, test_conf=None):
        if test_conf is not None:
            thresh = test_conf
        bboxes = self.box_coder.decode(anchors, bbox_pred, mode='xywht')
        bboxes = clip_boxes(bboxes, imgs)
        landmarks = self.box_coder.landmarkdecode(anchors, landmark_pred)
        landmarks = clip_landmarks(landmarks, imgs)
        scores = torch.max(cls_score, dim=2, keepdim=True)[0]
        keep = (scores >= thresh)[0, :, 0]
        if keep.sum() == 0:
            return [torch.zeros(1), torch.zeros(1), torch.zeros(1, 5), torch.zeros(1, 4)]
        scores = scores[:, keep, :]
        anchors = anchors[:, keep, :]
        cls_score = cls_score[:, keep, :]
        bboxes = bboxes[:, keep, :]
        landmarks = landmarks[:, keep, :]
        anchors_nms_idx = nms(torch.cat([bboxes, scores], dim=2)[0, :, :], nms_thresh)
        nms_scores, nms_class = cls_score[0, anchors_nms_idx, :].max(dim=1)
        output_boxes = torch.cat([
            bboxes[0, anchors_nms_idx, :],
            anchors[0, anchors_nms_idx, :]],
            dim=1
        )
        output_landmarks = landmarks[0, anchors_nms_idx, :]
        return [nms_scores, nms_class, output_boxes, output_landmarks]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
