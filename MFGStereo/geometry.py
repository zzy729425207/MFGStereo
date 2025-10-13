import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler


class Combined_Geo_Encoding_Volume:
    def __init__(self, geo_volume, mono_corr_volume, stereo_corr_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.geo_volume_pyramid = []
        self.mono_corr_volume_pyramid = []
        self.stereo_corr_volume_pyramid = []

        b, c, h, w, w1 = mono_corr_volume.shape # torch.Size([1, 8, 80, 160, 160])
        b, c, d, h, w = geo_volume.shape  # torch.Size([1, 8, 48, 80, 160])

        mono_corr_volume = mono_corr_volume.permute(0, 2, 3, 1, 4).reshape(b*h*w, c, 1, w1)
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)
        stereo_corr_volume = stereo_corr_volume.permute(0, 2, 3, 1, 4).reshape(b*h*w, 1, 1, w1)

        self.geo_volume_pyramid.append(geo_volume)
        self.mono_corr_volume_pyramid.append(mono_corr_volume)
        self.stereo_corr_volume_pyramid.append(stereo_corr_volume)

        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            mono_corr_volume = F.avg_pool2d(mono_corr_volume, [1,2], stride=[1,2])
            self.mono_corr_volume_pyramid.append(mono_corr_volume)

        for i in range(self.num_levels-1):
            stereo_corr_volume = F.avg_pool2d(stereo_corr_volume, [1,2], stride=[1,2])
            self.stereo_corr_volume_pyramid.append(stereo_corr_volume)

    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            mono_corr = self.mono_corr_volume_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            mono_corr = bilinear_sampler(mono_corr, init_coords_lvl)
            mono_corr = mono_corr.view(b, h, w, -1)

            stereo_corr = self.stereo_corr_volume_pyramid[i]
            stereo_corr = bilinear_sampler(stereo_corr,init_coords_lvl)
            stereo_corr = stereo_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(mono_corr)
            out_pyramid.append(stereo_corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap2, fmap3):
        B, D, H, W2 = fmap2.shape
        _, _, _, W3 = fmap3.shape
        fmap2_dtype = fmap2.dtype

        fmap2 = fmap2.view(B, D, H, W2)
        fmap3 = fmap3.view(B, D, H, W3)

        # a i j k: batch, feature, height, width
        # a i j h: batch, feature, height, disparity
        # a j k h: batch, height, width, disparity

        corr = torch.einsum('aijk,aijh->ajkh', fmap2, fmap3)
        corr = corr.reshape(B, 1, H, W2, W3).contiguous()
        return (corr / torch.sqrt(torch.tensor(D))).to(fmap2_dtype)