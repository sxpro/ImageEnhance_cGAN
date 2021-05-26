import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import pad_tensor, pad_tensor_back
import torch.nn.utils.spectral_norm as spectral_norm
        

class SpatialOffset(nn.Module):
    def __init__(self, ch_in, ks):
        super().__init__()

        nhidden = 128
        pw = ks // 2
        self.mlp_beta = nn.Sequential( 
                                nn.Conv2d(ch_in, nhidden, kernel_size=ks, padding=pw, padding_mode='reflect', bias=False), 
                                nn.LeakyReLU(0.1, inplace=True),
                                nn.Conv2d(nhidden, nhidden//2, kernel_size=ks, padding=pw, padding_mode='reflect', bias=False),
                                nn.LeakyReLU(0.1, inplace=True),
                                nn.Conv2d(nhidden//2, ch_in, kernel_size=ks, padding=pw, padding_mode='reflect', bias=False)
                                )
                                
    def forward(self, x):
        offset = self.mlp_beta(x)
        return offset
        

## normal version 2
class SpatialOffsetBlock(nn.Module):
    def __init__(self, ch_in, ch_ref, ks):
        super(SpatialOffsetBlock, self).__init__()
        
        nhidden = 64
        self.offset0 = SpatialOffset(ch_in, ks)
        self.norm_ref = nn.InstanceNorm2d(ch_ref, affine=False)

    def forward(self, x, ref):
        # x and residual should have the same height and width
        x_sigma = torch.std(x, dim=[2,3], keepdim=True)
        ref_norm = self.norm_ref(ref)
        offset = self.offset0(ref_norm)
        out = x + x_sigma*offset
        return out
        
        
class Condition(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(Condition, self).__init__()
        nhidden = 64
        self.conv = nn.Sequential(
                                nn.Conv2d(in_nc, nhidden, 6, stride=3, padding=3, padding_mode='reflect', groups=1, bias=True),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                nn.Conv2d(nhidden, nhidden, 4, stride=2,  padding=1, padding_mode='reflect', groups=1, bias=True),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                nn.Conv2d(nhidden, nf, 4, stride=2,  padding=1, padding_mode='reflect', groups=1, bias=True),
                                nn.LeakyReLU(0.02, inplace=True)
                                )
    def forward(self, x):
    
        if x.size()[2] > 600 or x.size()[3] > 600: # rescaling for computing condition code
            scale_factor = min(600./x.size()[2], 600./x.size()[3])
            x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear')
            print(f'Resize to {x.size()} with scale factor {scale_factor}')
        out = self.conv(x)
        return out
        
        
def ContrastAdjustment(x, scale0, scale1, midv):
    x_shift = x-midv
    zeros = torch.tensor([0.],device=x.device).expand_as(x)
    out = scale0 * torch.maximum(zeros, x_shift) + scale1 * torch.minimum(zeros, x_shift) + midv
    return out


class RetouchBlock(nn.Module):
    def __init__(self, in_nc, out_nc, base_nf=64, cond_nf=64, kernel_sizes=1):
        super(RetouchBlock, self).__init__()

        self.in_nc = in_nc
        self.base_nf = base_nf
        self.out_nc = out_nc
        nhidden = 64
        # intermediate layer
        self.conv0 = nn.Conv2d(in_nc, base_nf, kernel_sizes, 1, 0, padding_mode='reflect', bias=False)
        self.cond_scale0_0 = nn.Sequential( nn.Linear(cond_nf, nhidden,  bias=False),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Linear(nhidden, base_nf,  bias=False)
                                        )
        self.cond_scale0_1 = nn.Sequential( nn.Linear(cond_nf, nhidden,  bias=False),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Linear(nhidden, base_nf,  bias=False)
                                        )
        # self.cond_midv_0 = nn.Sequential( nn.Linear(cond_nf, nhidden,  bias=False),
                                        # nn.LeakyReLU(0.1, inplace=True),
                                        # nn.Linear(nhidden, 1,  bias=False)
                                        # )                      
        self.cond_brightness0 = nn.Sequential( nn.Linear(cond_nf, nhidden,  bias=False),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Linear(nhidden, base_nf,  bias=False)
                                        )
        # out layer
        self.conv1 = nn.Conv2d(base_nf, out_nc, kernel_sizes, 1, 0, padding_mode='reflect', bias=False)
        self.cond_scale1_0 = nn.Sequential( nn.Linear(cond_nf, nhidden,  bias=False),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Linear(nhidden, out_nc,  bias=False)
                                        )
        self.cond_scale1_1 = nn.Sequential( nn.Linear(cond_nf, nhidden,  bias=False),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Linear(nhidden, out_nc,  bias=False)
                                        )
        # self.cond_midv_1 = nn.Sequential( nn.Linear(cond_nf, nhidden,  bias=False),
                                        # nn.LeakyReLU(0.1, inplace=True),
                                        # nn.Linear(nhidden, 1,  bias=False)
                                        # )  
        self.cond_brightness1 = nn.Sequential( nn.Linear(cond_nf, nhidden,  bias=False),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Linear(nhidden, out_nc,  bias=False)
                                        )


    def forward(self, x, cond):
        ## 
        out = self.conv0(x)
        scale0_0 = self.cond_scale0_0(cond).view(-1, self.base_nf, 1, 1)
        scale0_1 = self.cond_scale0_1(cond).view(-1, self.base_nf, 1, 1)
        alpha0_0 = F.leaky_relu(1 + scale0_0, negative_slope=0.02)
        #alpha0_0 = 4 - F.leaky_relu(4 - alpha0_0, negative_slope=0.02)
        alpha0_1 = F.leaky_relu(1 + scale0_1, negative_slope=0.02)
        #alpha0_1 = 4 - F.leaky_relu(4 - alpha0_1, negative_slope=0.02)
        
        midv0 = torch.mean(out, dim=[2,3], keepdim=True)
        
        ##midv0 = self.cond_midv_0(cond).view(-1, 1, 1, 1)
        
        brightness0 = self.cond_brightness0(cond).view(-1, self.base_nf, 1, 1)
        
        out = ContrastAdjustment(out, alpha0_0, alpha0_1, midv0) + brightness0
        ##
        out = self.conv1(out)
        scale1_0 = self.cond_scale1_0(cond).view(-1, self.out_nc, 1, 1)
        scale1_1 = self.cond_scale1_1(cond).view(-1, self.out_nc, 1, 1)
        alpha1_0 = F.leaky_relu(1 + scale1_0, negative_slope=0.02)
        #alpha1_0 = 4 - F.leaky_relu(4 - alpha1_0, negative_slope=0.02)
        alpha1_1 = F.leaky_relu(1 + scale1_1, negative_slope=0.02)
        #alpha1_1 = 4 - F.leaky_relu(4 - alpha1_1, negative_slope=0.02)
        midv1 = torch.mean(out, dim=[2,3], keepdim=True)
        ##midv1 = self.cond_midv_1(cond).view(-1, 1, 1, 1)
        
        brightness1 = self.cond_brightness1(cond).view(-1, self.out_nc, 1, 1)
        
        out = ContrastAdjustment(out, alpha1_0, alpha1_1, midv1) + brightness1
        return out
        

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, nf=3, cond_nf=64, norm_layer=nn.InstanceNorm2d):
        super(Unet, self).__init__()
        self.downscale = 16
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nf = nf
        self.cond_nf = cond_nf
        
        # # merge
        self.merge_cond_mult = nn.Sequential( nn.Conv1d(cond_nf, cond_nf, 2, 1, 0,  bias=True),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Conv1d(cond_nf, cond_nf, 1, 1, 0, bias=True))
        self.merge_cond_offset = nn.Sequential( nn.Conv1d(cond_nf, cond_nf, 2, 1, 0,  bias=True),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Conv1d(cond_nf, cond_nf, 1, 1, 0, bias=True))
        self.merge_cond = nn.Linear(cond_nf, cond_nf,  bias=True)
        
        #size // 1 
        if self.nf!=self.in_ch:
            self.conv_in = nn.Conv2d(in_ch, nf, 1, 1, 0, bias=False)
        #size // 2
        self.down_conv_0 = nn.Conv2d(nf, nf*2, 1, 1, 0, padding_mode='reflect', bias=False)
        self.down_0 = nn.Conv2d(nf*2, nf*2, 4, 2, 1, groups=nf*2, padding_mode='reflect', bias=False)
        #size // 4
        self.down_conv_1 = nn.Conv2d(nf*2, nf*4, 1, 1, 0, padding_mode='reflect', bias=False)
        self.down_1 = nn.Conv2d(nf*4, nf*4, 4, 2, 1, groups=nf*4, padding_mode='reflect', bias=False)
        #size // 8
        self.down_conv_2 = nn.Conv2d(nf*4, nf*8, 1, 1, 0, padding_mode='reflect', bias=False)
        self.down_2 = nn.Conv2d(nf*8, nf*8, 4, 2, 1, groups=nf*8, padding_mode='reflect', bias=False)
        if self.downscale == 16:
            self.down_conv_3 = nn.Conv2d(nf*8, nf*16, 1, 1, 0, padding_mode='reflect', bias=False)
            self.down_3 = nn.Conv2d(nf*16, nf*16, 4, 2, 1, groups=nf*16, padding_mode='reflect', bias=False)
            #size // 8
            self.up_3 = nn.Conv2d(nf*16, nf*16, 3, 1, 1, padding_mode='reflect', groups=nf*16, bias=False)
            self.conv_up_3 = nn.Conv2d(nf*16, nf*8, 1, 1, 0, padding_mode='reflect', bias=False)
            self.modulate_3 = SpatialOffsetBlock(nf * 8, nf * 8, ks=3)
            self.retouch_3 = RetouchBlock(nf*8, nf*8, base_nf=cond_nf, cond_nf=cond_nf)
        #upsample
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        #size // 4
        self.up_2 = nn.Conv2d(nf*8, nf*8, 3, 1, 1, padding_mode='reflect', groups=nf*8, bias=False)
        self.conv_up_2 = nn.Conv2d(nf*8, nf*4, 1, 1, 0, padding_mode='reflect', bias=False)
        self.modulate_2 = SpatialOffsetBlock(nf*4, nf*4, ks=3)
        self.retouch_2 = RetouchBlock(nf*4, nf*4, base_nf=cond_nf, cond_nf=cond_nf)
        #size // 2
        self.up_1 = nn.Conv2d(nf*4, nf*4, 3, 1, 1, padding_mode='reflect', groups=nf*4, bias=False)
        self.conv_up_1 = nn.Conv2d(nf*4, nf*2, 1, 1, 0, padding_mode='reflect', bias=False)
        self.modulate_1 = SpatialOffsetBlock(nf * 2, nf * 2, ks=5)
        self.retouch_1 = RetouchBlock(nf*2, nf*2, base_nf=cond_nf, cond_nf=cond_nf)
        #size // 1
        self.up_0 = nn.Conv2d(nf*2, nf*2, 3, 1, 1, padding_mode='reflect', groups=nf*2, bias=False)
        self.conv_up_0 = nn.Conv2d(nf*2, nf*1, 1, 1, 0, padding_mode='reflect', bias=False)
        self.modulate_0 = SpatialOffsetBlock(nf * 1, nf * 1, ks=5)
        self.retouch_0 = RetouchBlock(nf*1, nf*1, base_nf=cond_nf, cond_nf=cond_nf)
        if self.nf!=self.out_ch:
            self.conv_out = nn.Conv2d(nf, out_ch, 1, 1, 0, bias=False)
    
    def forward(self, netC, x, ref):
        
        # merge condition map
        cond_x_code = torch.mean(netC(x), dim=[2,3], keepdim=False)
        cond_ref_code = torch.mean(netC(ref), dim=[2,3], keepdim=False)
        cond_stack = torch.stack([cond_x_code, cond_ref_code-cond_x_code],dim=2)
        cond_code_offset = self.merge_cond_offset(cond_stack).squeeze(2)
        cond_code_mult = F.relu(self.merge_cond_mult(cond_stack)).squeeze(2)
        cond_retouch_code = self.merge_cond( cond_x_code*cond_code_mult + cond_code_offset)
        
        # padding
        x, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x, divide=self.downscale)
        
        # in conv 
        if self.nf!=self.in_ch:
            x0 = self.conv_in(x)
        else:
            x0 = x
        #scale // 2
        x1 = self.down_0(self.down_conv_0(x0))
        #scale // 4
        x2 = self.down_1(self.down_conv_1(x1))
        #scale //8
        x3 = self.down_2(self.down_conv_2(x2))
        if self.downscale == 16:
            #scale //16
            x4 = self.down_3(self.down_conv_3(x3))
            #scale //8
            up_x3 = self.conv_up_3(self.up_3(self.up(x4)))
            up_x3 = self.modulate_3(up_x3, x3)
            up_x3 = self.retouch_3(up_x3, cond_retouch_code)
        else: 
            up_x3 = x3
        #scale // 4
        up_x2 = self.conv_up_2(self.up_2(self.up(up_x3)))
        up_x2 = self.modulate_2(up_x2, x2)
        up_x2 = self.retouch_2(up_x2, cond_retouch_code)
        #scale // 2
        up_x1 = self.conv_up_1(self.up_1(self.up(up_x2)))
        up_x1 = self.modulate_1(up_x1, x1)
        up_x1 = self.retouch_1(up_x1, cond_retouch_code)
        #scale // 1
        up_x0 = self.conv_up_0(self.up_0(self.up(up_x1)))
        up_x0 = self.modulate_0(up_x0, x0)
        up_x0 = self.retouch_0(up_x0, cond_retouch_code)
        # out conv
        if self.nf!=self.in_ch:
            out = self.conv_out(up_x0)
        else: out = up_x0
        
        out = pad_tensor_back(out, pad_left, pad_right, pad_top, pad_bottom)
        return out


# Ablation study 1: Simple cat+linear for code merge
class UnetSimpleCondMerge(nn.Module):
    def __init__(self, in_ch, out_ch, nf=3, cond_nf=64, norm_layer=nn.InstanceNorm2d):
        super(UnetSimpleCondMerge, self).__init__()
        self.downscale = 16
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nf = nf
        self.cond_nf = cond_nf
        
        # # merge
        self.merge_cond = nn.Sequential( nn.Conv1d(cond_nf, cond_nf, 2, 1, 0,  bias=True),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Conv1d(cond_nf, cond_nf, 1, 1, 0, bias=True),
                                        nn.LeakyReLU(0.1, inplace=True))
        
        #size // 1 
        if self.nf!=self.in_ch:
            self.conv_in = nn.Conv2d(in_ch, nf, 1, 1, 0, bias=False)
        #size // 2
        self.down_conv_0 = nn.Conv2d(nf, nf*2, 1, 1, 0, padding_mode='reflect', bias=False)
        self.down_0 = nn.Conv2d(nf*2, nf*2, 4, 2, 1, groups=nf*2, padding_mode='reflect', bias=False)
        #size // 4
        self.down_conv_1 = nn.Conv2d(nf*2, nf*4, 1, 1, 0, padding_mode='reflect', bias=False)
        self.down_1 = nn.Conv2d(nf*4, nf*4, 4, 2, 1, groups=nf*4, padding_mode='reflect', bias=False)
        #size // 8
        self.down_conv_2 = nn.Conv2d(nf*4, nf*8, 1, 1, 0, padding_mode='reflect', bias=False)
        self.down_2 = nn.Conv2d(nf*8, nf*8, 4, 2, 1, groups=nf*8, padding_mode='reflect', bias=False)
        if self.downscale == 16:
            self.down_conv_3 = nn.Conv2d(nf*8, nf*16, 1, 1, 0, padding_mode='reflect', bias=False)
            self.down_3 = nn.Conv2d(nf*16, nf*16, 4, 2, 1, groups=nf*16, padding_mode='reflect', bias=False)
            #size // 8
            self.up_3 = nn.Conv2d(nf*16, nf*16, 3, 1, 1, padding_mode='reflect', groups=nf*16, bias=False)
            self.conv_up_3 = nn.Conv2d(nf*16, nf*8, 1, 1, 0, padding_mode='reflect', bias=False)
            self.modulate_3 = SpatialOffsetBlock(nf * 8, nf * 8, ks=3)
            self.retouch_3 = RetouchBlock(nf*8, nf*8, base_nf=cond_nf, cond_nf=cond_nf)
        #upsample
        self.up = nn.Upsample(scale_factor=2, mode='nearest') 
        #size // 4
        self.up_2 = nn.Conv2d(nf*8, nf*8, 3, 1, 1, padding_mode='reflect', groups=nf*8, bias=False)
        self.conv_up_2 = nn.Conv2d(nf*8, nf*4, 1, 1, 0, padding_mode='reflect', bias=False)
        self.modulate_2 = SpatialOffsetBlock(nf*4, nf*4, ks=3)
        self.retouch_2 = RetouchBlock(nf*4, nf*4, base_nf=cond_nf, cond_nf=cond_nf)
        #size // 2
        self.up_1 = nn.Conv2d(nf*4, nf*4, 3, 1, 1, padding_mode='reflect', groups=nf*4, bias=False)
        self.conv_up_1 = nn.Conv2d(nf*4, nf*2, 1, 1, 0, padding_mode='reflect', bias=False)
        self.modulate_1 = SpatialOffsetBlock(nf * 2, nf * 2, ks=5)
        self.retouch_1 = RetouchBlock(nf*2, nf*2, base_nf=cond_nf, cond_nf=cond_nf)
        #size // 1
        self.up_0 = nn.Conv2d(nf*2, nf*2, 3, 1, 1, padding_mode='reflect', groups=nf*2, bias=False)
        self.conv_up_0 = nn.Conv2d(nf*2, nf*1, 1, 1, 0, padding_mode='reflect', bias=False)
        self.modulate_0 = SpatialOffsetBlock(nf * 1, nf * 1, ks=5)
        self.retouch_0 = RetouchBlock(nf*1, nf*1, base_nf=cond_nf, cond_nf=cond_nf)
        if self.nf!=self.out_ch:
            self.conv_out = nn.Conv2d(nf, out_ch, 1, 1, 0, bias=False)
    
    def forward(self, netC, x, ref):
        
        # merge condition map
        cond_x_code = torch.mean(netC(x), dim=[2,3], keepdim=False)
        cond_ref_code = torch.mean(netC(ref), dim=[2,3], keepdim=False)
        cond_stack = torch.stack([cond_x_code, cond_ref_code],dim=2)
        cond_retouch_code = self.merge_cond(cond_stack).squeeze(2)
        
        # padding
        x, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x, divide=self.downscale)
        
        # in conv 
        if self.nf!=self.in_ch:
            x0 = self.conv_in(x)
        else:
            x0 = x
        #scale // 2
        x1 = self.down_0(self.down_conv_0(x0))
        #scale // 4
        x2 = self.down_1(self.down_conv_1(x1))
        #scale //8
        x3 = self.down_2(self.down_conv_2(x2))
        if self.downscale == 16:
            #scale //16
            x4 = self.down_3(self.down_conv_3(x3))
            #scale //8
            up_x3 = self.conv_up_3(self.up_3(self.up(x4)))
            up_x3 = self.modulate_3(up_x3, x3)
            up_x3 = self.retouch_3(up_x3, cond_retouch_code)
        else: 
            up_x3 = x3
        #scale // 4
        up_x2 = self.conv_up_2(self.up_2(self.up(up_x3)))
        up_x2 = self.modulate_2(up_x2, x2)
        up_x2 = self.retouch_2(up_x2, cond_retouch_code)
        #scale // 2
        up_x1 = self.conv_up_1(self.up_1(self.up(up_x2)))
        up_x1 = self.modulate_1(up_x1, x1)
        up_x1 = self.retouch_1(up_x1, cond_retouch_code)
        #scale // 1
        up_x0 = self.conv_up_0(self.up_0(self.up(up_x1)))
        up_x0 = self.modulate_0(up_x0, x0)
        up_x0 = self.retouch_0(up_x0, cond_retouch_code)
        # out conv
        if self.nf!=self.in_ch:
            out = self.conv_out(up_x0)
        else: out = up_x0
        
        out = pad_tensor_back(out, pad_left, pad_right, pad_top, pad_bottom)
        return out
