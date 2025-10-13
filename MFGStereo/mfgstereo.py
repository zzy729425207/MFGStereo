import torch
import torch.nn as nn
import torch.nn.functional as F
from update import BasicMultiUpdateBlock
from extractor import MultiBasicEncoder, Feature
from geometry import Combined_Geo_Encoding_Volume
from submodule import *
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2_decoder
import time

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv3 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 6, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv3_up = BasicConv(in_channels * 6, in_channels * 4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels * 2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(
            BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1), )

        self.agg_1 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.feature_att_8 = FeatureAtt(in_channels * 2, 64)
        self.feature_att_16 = FeatureAtt(in_channels * 4, 192)
        self.feature_att_32 = FeatureAtt(in_channels * 6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels * 4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels * 2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv


class Feat_transfer(nn.Module):
    def __init__(self, dim_list):
        super(Feat_transfer, self).__init__()
        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=int(48 + dim_list[0]), out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(48), nn.ReLU()
        )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=int(64 + dim_list[0]), out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
        )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=int(192 + dim_list[0]), out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(192), nn.ReLU()
        )
        self.conv32x = nn.Sequential(
            nn.Conv2d(in_channels=dim_list[0], out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(160), nn.ReLU()
        )
        self.conv_up_32x = nn.ConvTranspose2d(160,
                                              192,
                                              kernel_size=3,
                                              padding=1,
                                              output_padding=1,
                                              stride=2,
                                              bias=False)
        self.conv_up_16x = nn.ConvTranspose2d(192,
                                              64,
                                              kernel_size=3,
                                              padding=1,
                                              output_padding=1,
                                              stride=2,
                                              bias=False)
        self.conv_up_8x = nn.ConvTranspose2d(64,
                                             48,
                                             kernel_size=3,
                                             padding=1,
                                             output_padding=1,
                                             stride=2,
                                             bias=False)

        self.res_16x = nn.Conv2d(dim_list[0], 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0], 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0], 48, kernel_size=1, padding=0, stride=1)

    def forward(self, features):
        features_mono_list = []
        feat_32x = self.conv32x(features[3])
        feat_32x_up = self.conv_up_32x(feat_32x)
        feat_16x = self.conv16x(torch.cat((features[2], feat_32x_up), 1)) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)
        features_mono_list.append(feat_32x)
        return features_mono_list


class Feat_transfer_cnet(nn.Module):
    def __init__(self, dim_list, output_dim):
        super(Feat_transfer_cnet, self).__init__()

        self.res_16x = nn.Conv2d(dim_list[0] + 192, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0] + 96, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0] + 48, output_dim, kernel_size=3, padding=1, stride=1)

    def forward(self, features, stem_x_list):
        features_list = []
        feat_16x = self.res_16x(torch.cat((features[2], stem_x_list[0]), 1))
        feat_8x = self.res_8x(torch.cat((features[1], stem_x_list[1]), 1))
        feat_4x = self.res_4x(torch.cat((features[0], stem_x_list[2]), 1))
        features_list.append([feat_4x, feat_4x])
        features_list.append([feat_8x, feat_8x])
        features_list.append([feat_16x, feat_16x])
        return features_list


import math


def estimate_confidence(prob):  #
    _, D, _, _ = prob.shape
    # _, _, D, _, _ = corr_volume.shape
    # prob = F.softmax(corr_volume.squeeze(1), dim=1)  # B, 1, H, W2, W3 -> B, H, W2, W3
    # conf_left = logsumexp_eps * torch.logsumexp(prob_left/logsumexp_eps, dim=3, keepdim=False) # B H W2
    # Alternative based on information entropy
    conf_left = -torch.sum(prob * torch.log2(prob + 1e-6), dim=1, keepdim=False) / math.log2(D)  # B H W2
    conf_left = 1 - conf_left  # High confidence for low entropy 越接近某个，熵越低
    return conf_left.unsqueeze(1)  # B 1 H W2


def disp_warping(disp, img, right_disp=False):
    B, _, H, W = disp.shape

    mycoords_y, mycoords_x = torch.meshgrid(torch.arange(H, dtype=disp.dtype, device=disp.device),
                                            torch.arange(W, dtype=disp.dtype, device=disp.device), indexing='ij')
    mycoords_x = mycoords_x[None].repeat(B, 1, 1).to(disp.device)
    mycoords_y = mycoords_y[None].repeat(B, 1, 1).to(disp.device)

    if right_disp:
        grid = 2 * torch.cat([(mycoords_x + disp.squeeze(1)).unsqueeze(-1) / W, mycoords_y.unsqueeze(-1) / H], -1) - 1
    else:
        grid = 2 * torch.cat([(mycoords_x - disp.squeeze(1)).unsqueeze(-1) / W, mycoords_y.unsqueeze(-1) / H], -1) - 1

    # grid_sample: B,C,H,W & B H W 2 -> B C H W
    warped_img = F.grid_sample(img, grid, align_corners=True)

    return warped_img


def softlrc(disp2, disp3, lrc_th=1.0):
    div_const = math.log(1 + math.exp(lrc_th))

    warped_disp2 = disp_warping(F.relu(disp3), disp2, right_disp=True)  # B 1 H W  # 向对面warp
    warped_disp3 = disp_warping(F.relu(disp2), disp3, right_disp=False)  # B 1 H W  # 向对面warp

    softlrc_disp2 = F.softplus(-torch.abs(disp2 - warped_disp3) + lrc_th) / div_const  # lrc weights in (0,1)  #B 1 H W
    softlrc_disp3 = F.softplus(-torch.abs(disp3 - warped_disp2) + lrc_th) / div_const  # lrc weights in (0,1) #B 1 H W

    return softlrc_disp2, softlrc_disp3


def weighted_lsq(mde, disp, conf, min_quantile=0.2, max_quantile=0.9):
    B, _, _, _ = mde.shape
    mde_dtype = mde.dtype

    # Weighted LSQ
    mde, disp, conf = mde.reshape(B, -1).float(), disp.reshape(B, -1).float(), conf.reshape(B, -1).float()

    disp = F.relu(disp)

    scale_shift = torch.zeros((B, 2), device=mde.device)

    for b in range(B):
        _mono = mde[b].unsqueeze(0)
        _stereo = disp[b].unsqueeze(0)
        _conf = conf[b].unsqueeze(0)

        _min_disp = torch.quantile(_stereo.flatten(), min_quantile)
        _max_disp = torch.quantile(_stereo.flatten(), max_quantile)

        _quantile_mask = (_min_disp <= _stereo) & (_stereo <= _max_disp)

        _mono = _mono[_quantile_mask].unsqueeze(0)
        _conf = _conf[_quantile_mask].unsqueeze(0)
        _stereo = _stereo[_quantile_mask].unsqueeze(0)

        _mono = torch.abs(_mono.flatten().unsqueeze(0))
        _stereo = torch.abs(_stereo.flatten().unsqueeze(0))
        _conf = torch.abs(_conf.flatten().unsqueeze(0))

        _conf = _conf * (1 - 0.1) + 0.1

        weights = torch.sqrt(_conf)
        A_matrix = _mono * weights
        A_matrix = torch.cat([A_matrix.unsqueeze(-1), weights.unsqueeze(-1)], -1)
        B_matrix = (_stereo * weights).unsqueeze(-1)

        _scale_shift = torch.linalg.lstsq(A_matrix, B_matrix)[0].squeeze(2)  # 1 x 2 x 1 -> 1 x 2,
        scale_shift[b] = _scale_shift.squeeze(0)

    return scale_shift[:, 0:1].reshape(B, 1, 1, 1).to(mde_dtype), scale_shift[:, 1:2].reshape(B, 1, 1, 1).to(mde_dtype)


def weighted_lsq_outconf(mde, disp, min_quantile=0.2, max_quantile=0.9):
    B, _, _, _ = mde.shape
    mde_dtype = mde.dtype

    # Weighted LSQ
    mde, disp = mde.reshape(B, -1).float(), disp.reshape(B, -1).float()

    disp = F.relu(disp)

    scale_shift = torch.zeros((B, 2), device=mde.device)

    for b in range(B):
        _mono = mde[b].unsqueeze(0)
        _stereo = disp[b].unsqueeze(0)

        _min_disp = torch.quantile(_stereo.flatten(), min_quantile)
        _max_disp = torch.quantile(_stereo.flatten(), max_quantile)

        _quantile_mask = (_min_disp <= _stereo) & (_stereo <= _max_disp)

        _mono = _mono[_quantile_mask].unsqueeze(0)
        _stereo = _stereo[_quantile_mask].unsqueeze(0)

        _mono = torch.abs(_mono.flatten().unsqueeze(0))
        _stereo = torch.abs(_stereo.flatten().unsqueeze(0))

        A_matrix = torch.cat([
            _mono.unsqueeze(-1),
            torch.ones_like(_mono).unsqueeze(-1)
        ], -1)  # [1, N, 2]
        B_matrix = _stereo.unsqueeze(-1)

        _scale_shift = torch.linalg.lstsq(A_matrix, B_matrix)[0].squeeze(2)  # 1 x 2 x 1 -> 1 x 2,
        scale_shift[b] = _scale_shift.squeeze(0)

    return scale_shift[:, 0:1].reshape(B, 1, 1, 1).to(mde_dtype), scale_shift[:, 1:2].reshape(B, 1, 1, 1).to(mde_dtype)


def fuzzy_and(x, y):
    return x * y


def fuzzy_or(x, y):
    return x + y - x * y


def fuzzy_not(x):
    return 1 - x


def fuzzy_and_zadeh(x, y, eps=1e-3):
    return -eps * torch.logsumexp(-torch.cat([x, y], 1) / eps, 1, keepdim=True)


def fuzzy_or_zadeh(x, y, eps=1e-3):
    return eps * torch.logsumexp(torch.cat([x, y], 1) / eps, 1, keepdim=True)


def handcrafted_mirror_detector(stereo_disp, mono_disp, stereo_conf, mono_conf, conf_th=0.5, step_gain=20):
    # Handcrafted confidence: (MONO >> STEREO AND LRC_MONO) OR (LRC_MONO AND ~LRC_STEREO)
    # Four cases:
    # BOTH LRCs are bad (LRC_MONO=0; LRC_STEREO=0): we are in occlusions where mono is typically better than stereo, but if scale is wrong mono is not trustable
    # LRC_STEREO is bad (LRC_MONO=1; LRC_STEREO=0): probably there are high-frequency details better captured by mono
    # LRC_MONO is bad (LRC_MONO=0; LRC_STEREO=1): probably there is an optical illusion in the stereo pair or mono is not consistent with the stereo pair
    # BOTH LRCs are good (LRC_MONO=1; LRC_STEREO=1): probably the stereo pair is consistent with the mono prediction: usually stereo is better here, however, mono is predicting high disparity values probably there is a mirror

    mono_and_stereo_conf = fuzzy_and(stereo_conf, mono_conf)
    mono_near_wrt_stereo = F.sigmoid(step_gain * (mono_disp - stereo_disp))  # 提取出两个置信度不一致的位置
    mono_is_better_a = fuzzy_and(mono_and_stereo_conf, mono_near_wrt_stereo)  # 双目可靠且单目可靠
    mono_is_better_b = fuzzy_and(fuzzy_not(stereo_conf), mono_conf)  # 双目不可靠，单目可靠
    mono_is_better = fuzzy_or(mono_is_better_a, mono_is_better_b)  # 并起来，单目可靠的位置

    return F.sigmoid(step_gain * (mono_is_better - conf_th))  # 选出mono可信的位置，stereo不可信的位置


def truncate_corr_volume_v2(disp_left, conf_left, attenuation_gain=0.1):
    B, _, H, W = disp_left.shape

    disp_values = torch.arange(0, W, dtype=disp_left.dtype, device=disp_left.device)

    disp_values_left = disp_values.view(1, 1, 1, -1).repeat(B, 1, 1, 1)

    mycoords_y, mycoords_x = torch.meshgrid(
        torch.arange(H, dtype=disp_left.dtype, device=disp_left.device),
        torch.arange(W, dtype=disp_left.dtype, device=disp_left.device),
        indexing='ij'  # 确保坐标与图像维度对应 (y, x)
    )

    mycoords_x = mycoords_x[None].repeat(B, 1, 1).to(disp_left.device)  # x坐标（图像列方向）
    mycoords_y = mycoords_y[None].repeat(B, 1, 1).to(disp_left.device)  # y坐标（图像行方向）

    truncate_cost_position = (mycoords_x.unsqueeze(1) - disp_left)

    prob = F.sigmoid(truncate_cost_position)

    truncate_corr_left = 1 * (1 - conf_left) + (conf_left) * (prob * (1 - attenuation_gain) + attenuation_gain)

    return truncate_corr_left


def truncate_corr_volume(disp_left):
    B, _, H, W = disp_left.shape
    disp_values = torch.arange(0, W, dtype=disp_left.dtype, device=disp_left.device)
    disp_values_left = disp_values.view(1, -1).repeat(B, 1)  # B W/4

    mycoords_y, mycoords_x = torch.meshgrid(torch.arange(W, dtype=disp_left.dtype, device=disp_left.device),
                                            torch.arange(W, dtype=disp_left.dtype, device=disp_left.device),
                                            indexing='ij')
    mycoords_x = mycoords_x[None].repeat(B, 1, 1).to(disp_left.device)  # 1 160 160

    truncate_center = 1 - F.sigmoid(20 * (mycoords_x - disp_values_left.unsqueeze(2)))
    # truncate_center = truncate_center[None].repeat(B,1,1,1).to(disp_left.device)

    return truncate_center.unsqueeze(1).unsqueeze(1)


class MFGStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args.encoder = "vitl"
        self.args.hidden_dims = [128] * 3
        self.args.corr_implementation = "reg"
        self.args.corr_levels = 2
        self.args.corr_radius = 4
        self.args.n_downsample = 2
        self.args.n_gru_layers = 3
        self.args.max_disp = 192
        self.args.confidence = True
        context_dims = self.args.hidden_dims
        self.args.dpav2_path = r"F:\单双目立体匹配\OurStereo\pretrained"

        self.register_buffer('mean', torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255)
        self.register_buffer('std', torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255)

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }

        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        dim_list_ = mono_model_configs[self.args.encoder]['features']
        dim_list = []
        dim_list.append(dim_list_)
        self.feat_transfer = Feat_transfer(dim_list)
        self.feat_transfer_cnet = Feat_transfer_cnet(dim_list, output_dim=self.args.hidden_dims[0])

        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=self.args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], self.args.hidden_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.args.n_gru_layers)])

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )
        self.stem_8 = nn.Sequential(
            BasicConv_IN(48, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96), nn.ReLU()
        )
        self.stem_16 = nn.Sequential(
            BasicConv_IN(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(192), nn.ReLU()
        )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
        )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        self.OGDDMEncoder = OGDDMEncoder()

        depth_anything = DepthAnythingV2(**mono_model_configs[self.args.encoder])
        depth_anything_decoder = DepthAnythingV2_decoder(
            **mono_model_configs[args.encoder])  # F:\单双目立体匹配\MonSter-main\pretrained
        state_dict_dpt = torch.load(f'{self.args.dpav2_path}/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
        depth_anything.load_state_dict(state_dict_dpt, strict=True)
        depth_anything_decoder.load_state_dict(state_dict_dpt, strict=False)
        self.mono_encoder = depth_anything.pretrained
        self.mono_decoder = depth_anything.depth_head
        self.feat_decoder = depth_anything_decoder.depth_head
        self.mono_encoder.requires_grad_(False)  # 这里可以把更新更新设置为False
        self.mono_decoder.requires_grad_(False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision,
                      dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp * 4., spx_pred).unsqueeze(1)

        return up_disp

    def infer_mono(self, image1, image2):
        height_ori, width_ori = image1.shape[2:]
        resize_image1 = F.interpolate(image1, scale_factor=14 / 16, mode='bilinear', align_corners=True)
        resize_image2 = F.interpolate(image2, scale_factor=14 / 16, mode='bilinear', align_corners=True)

        patch_h, patch_w = resize_image1.shape[-2] // 14, resize_image1.shape[-1] // 14
        features_left_encoder = self.mono_encoder.get_intermediate_layers(resize_image1, self.intermediate_layer_idx[
            self.args.encoder], return_class_token=True)
        features_right_encoder = self.mono_encoder.get_intermediate_layers(resize_image2, self.intermediate_layer_idx[
            self.args.encoder], return_class_token=True)
        depth_mono = self.mono_decoder(features_left_encoder, patch_h, patch_w)
        depth_mono = F.relu(depth_mono)

        depth_mono_right = self.mono_decoder(features_right_encoder, patch_h, patch_w)
        depth_mono_right = F.relu(depth_mono_right)

        depth_mono = F.interpolate(depth_mono, size=(height_ori, width_ori), mode='bilinear', align_corners=False)
        depth_mono_right = F.interpolate(depth_mono_right, size=(height_ori, width_ori), mode='bilinear',
                                         align_corners=False)

        features_left_4x, features_left_8x, features_left_16x, features_left_32x = self.feat_decoder(
            features_left_encoder, patch_h, patch_w)
        features_right_4x, features_right_8x, features_right_16x, features_right_32x = self.feat_decoder(
            features_right_encoder, patch_h, patch_w)

        return depth_mono, depth_mono_right, [features_left_4x, features_left_8x, features_left_16x,
                                              features_left_32x], [
                   features_right_4x, features_right_8x, features_right_16x, features_right_32x]

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        image1 = ((image1 - self.mean) / self.std).contiguous()
        image2 = ((image2 - self.mean) / self.std).contiguous()

        with autocast(enabled=self.args.mixed_precision):
            depth_mono_left, depth_mono_right, features_mono_left, features_mono_right = self.infer_mono(image1, image2)

            scale_factor = 1 / (self.args.n_downsample ** 2)
            size = (int(depth_mono_left.shape[-2] * scale_factor), int(depth_mono_left.shape[-1] * scale_factor))

            disp_mono_4x = F.interpolate(depth_mono_left, size=size, mode='bilinear', align_corners=False)

            disp_mono_4y = F.interpolate(depth_mono_right, size=size, mode='bilinear', align_corners=False)

            features_left = self.feat_transfer(features_mono_left)
            features_right = self.feat_transfer(features_mono_right)

            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_8x = self.stem_8(stem_4x)
            stem_16x = self.stem_16(stem_8x)

            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)

            stem_x_list = [stem_16x, stem_8x, stem_4x]

            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))

            gwc_volume_left = build_gwc_volume(match_left, match_right, self.args.max_disp // 4, 8)
            gwc_volume_left = self.corr_stem(gwc_volume_left)
            gwc_volume_left = self.corr_feature_att(gwc_volume_left, features_left[0])
            geo_encoding_volume_left = self.cost_agg(gwc_volume_left, features_left)  # torch.Size([1, 8, 48, 80, 160])

            # Init disp from geometry encoding volume
            prob_left = F.softmax(self.classifier(geo_encoding_volume_left).squeeze(1), dim=1)
            init_disp_left = disparity_regression(prob_left, self.args.max_disp // 4)

            gwc_volume_right = build_gwc_volume_right(match_right, match_left, self.args.max_disp // 4, 8)
            gwc_volume_right = self.corr_stem(gwc_volume_right)
            gwc_volume_right = self.corr_feature_att(gwc_volume_right, features_right[0])
            geo_encoding_volume_right = self.cost_agg(gwc_volume_right, features_right)
            prob_right = F.softmax(self.classifier(geo_encoding_volume_right).squeeze(1), dim=1)
            init_disp_right = disparity_regression(prob_right, self.args.max_disp // 4)

            corr_volume_mono = 1.73 * build_corr_from_mono(disp_mono_4x, disp_mono_4y, self.OGDDMEncoder)

            if self.args.confidence:
                # 通过prob的混乱程度判断置信度
                coarse_left_conf = estimate_confidence(prob_left)  # 通过熵的混乱程度预测置信度 CL
                coarse_right_conf = estimate_confidence(prob_right)  # 通过熵的混乱程度预测置信度 CR

                # 通过warp之后的差值大小判断置信度
                softLRC_left, softLRC_right = softlrc(init_disp_left, init_disp_right, 1)  # 可靠的双目预测  LRCC

                coarse_dispmonoconf2_lowres = coarse_left_conf * softLRC_left  # 最终置信度
                coarse_dispmonoconf3_lowres = coarse_right_conf * softLRC_right  # 最终置信度

                global_scale_left, global_shift_left = weighted_lsq(torch.cat([disp_mono_4x, disp_mono_4y], 1),
                                                                torch.cat([init_disp_left, init_disp_right], 1),
                                                                torch.cat(
                                                                    [coarse_dispmonoconf2_lowres,
                                                                     coarse_dispmonoconf3_lowres], 1))

                global_scale_right, global_shift_right = global_scale_left, global_shift_left

                init_disp_left_scale = torch.sum(global_scale_left * disp_mono_4x + global_shift_left, dim=1,
                                                 keepdim=True)  # 低尺度的
                init_disp_right_scale = torch.sum(global_scale_right * disp_mono_4y + global_shift_right, dim=1,
                                                  keepdim=True)

                softlrc_coarse_scaled_mde2_lowres, _ = softlrc(init_disp_left_scale, init_disp_right_scale, 1)

                mde2_mirrorconf_lowres = handcrafted_mirror_detector(init_disp_left, init_disp_left_scale,
                                                                     coarse_dispmonoconf2_lowres,
                                                                     softlrc_coarse_scaled_mde2_lowres,
                                                                     conf_th=0.98)

                mono_mismatch_mask = truncate_corr_volume(init_disp_left_scale)

                geo_encoding_volume_left = (1 - mde2_mirrorconf_lowres.unsqueeze(2)) * geo_encoding_volume_left

                corr_volume_mono = mono_mismatch_mask * corr_volume_mono

            else:
                bs, _, _, _ = init_disp_left.shape
                for i in range(bs):
                    with autocast(enabled=self.args.mixed_precision):
                        global_scale_left, global_shift_left = weighted_lsq_outconf(torch.cat([disp_mono_4x, disp_mono_4y], 1),
                                                                            torch.cat([init_disp_left, init_disp_right],
                                                                                      1))
                        init_disp_left_scale = torch.sum(global_scale_left * disp_mono_4x + global_shift_left, dim=1,
                                                         keepdim=True)  # 低尺度的

            del prob_left, gwc_volume_left, prob_right, gwc_volume_right

            if not test_mode:
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

                xspy = self.spx_4(features_right[0])
                xspy = self.spx_2(xspy, stem_2y)
                spy_pred = self.spx(xspy)
                spy_pred = F.softmax(spy_pred, 1)

            cnet_list = self.feat_transfer_cnet(features_mono_left, stem_x_list)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                        zip(inp_list, self.context_zqr_convs)]

        geo_block = Combined_Geo_Encoding_Volume

        stereo_corr_volume = mono_mismatch_mask * Combined_Geo_Encoding_Volume.corr(match_left, match_left)

        geo_fn = geo_block(geo_encoding_volume_left.float(), corr_volume_mono.float(), stereo_corr_volume.float(),
                           radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp_left_scale
        disp_preds = []

        # GRUs iterations to update disparity
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision):
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp,
                                                                      iter16=self.args.n_gru_layers == 3,
                                                                      iter08=self.args.n_gru_layers >= 2)

            disp = disp + delta_disp
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        init_disp_left = context_upsample(init_disp_left * 4., spx_pred.float()).unsqueeze(1)
        init_disp_right = context_upsample(init_disp_right * 4., spy_pred.float()).unsqueeze(1)

        return init_disp_left, init_disp_right, disp_preds
