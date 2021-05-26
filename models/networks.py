import torch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.unet_arch as unet_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'unet':
        netG = unet_arch.Unet(in_ch=opt_net['in_nc'], out_ch=opt_net['out_nc'], nf=opt_net['base_nf'],
                                 cond_nf=opt['condition_nf'])
    elif which_model == 'unet_simple_merge':
        netG = unet_arch.UnetSimpleCondMerge(in_ch=opt_net['in_nc'], out_ch=opt_net['out_nc'], nf=opt_net['base_nf'],
                                 cond_nf=opt['condition_nf'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
# Condition Network
def define_C(opt):
    opt_net = opt['network_C']
    which_model = opt_net['which_model_C']
    if which_model == 'CondNet':
        netC = unet_arch.Condition(in_nc=opt_net['in_nc'], nf=opt['condition_nf'])
    else:
        raise NotImplementedError('Condition model [{:s}] not recognized'.format(which_model))
    return netC

# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'patchgan':
        if opt['gan_type'] in ['gan','ragan']:
            netD = NLayerDiscriminator(input_nc=opt['condition_nf'], ndf=opt_net['nf'], use_sigmoid=False)
        else:
            netD = NLayerDiscriminator(input_nc=opt['condition_nf'], ndf=opt_net['nf'], use_sigmoid=True)
    elif which_model == 'vectorgan':
        if opt['gan_type'] in ['gan','ragan']:
            netD = VectorDiscriminator(input_nc=opt['condition_nf'], use_sigmoid=False)
        else:
            netD = VectorDiscriminator(input_nc=opt['condition_nf'], use_sigmoid=True)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD

# Discriminator
def define_D_pair(opt):
    opt_net = opt['network_D_pair']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'patchgan':
        netD = NLayerDiscriminator(input_nc=opt_net['in_nc'], ndf=opt_net['nf'])
    elif which_model == 'vectorgan':
        if opt['gan_type'] in ['gan','ragan']:
            netD = VectorDiscriminator(input_nc=2*opt['condition_nf'], use_sigmoid=False)
        else:
            netD = VectorDiscriminator(input_nc=2*opt['condition_nf'], use_sigmoid=True)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD
    
# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF

# Defines the PatchGAN discriminator with the specified arguments.
import torch.nn as nn
import numpy as np
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=32, n_layers=2, norm_layer=torch.nn.BatchNorm2d, use_sigmoid=True, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, padding_mode='reflect'),
            nn.LeakyReLU(0.1, True)
        ]
        
        nf_div = 1
        nf_div_prev = 1
        for n in range(1, n_layers):
            nf_div_prev = nf_div
            nf_div = min(2**n, 16)
            sequence += [
                nn.Conv2d(ndf//nf_div_prev, ndf//nf_div,
                          kernel_size=kw, stride=2, padding=padw, padding_mode='reflect'),
                nn.LeakyReLU(0.1, True)
            ]
            
        nf_div_prev = nf_div
        nf_div = min(2**n_layers, 16)
        sequence += [
                nn.Conv2d(ndf//nf_div_prev, ndf//nf_div,
                      kernel_size=kw, stride=2, padding=padw, padding_mode='reflect'),
                ]
                
        sequence1 = [
                norm_layer(ndf//nf_div),
                nn.LeakyReLU(0.1, False)
                ]
        sequence2 = [
                nn.LeakyReLU(0.1, False)
                ]
        
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model_shared = nn.Sequential(*sequence)
        self.model_split1 = nn.Sequential(*sequence1)
        self.model_split2 = nn.Sequential(*sequence2)
        self.model_out = nn.Conv2d(2*ndf//nf_div, 1, kernel_size=kw, stride=1, padding=padw, padding_mode='reflect')

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        mid_layer = self.model_shared(input)
        mid_layer1 = self.model_split1(mid_layer)
        mid_layer2 = self.model_split2(mid_layer)
        stack_layers = torch.cat([mid_layer1, mid_layer2], dim=1)
        return torch.mean(self.model_out(stack_layers), dim=[2,3])


class VectorDiscriminator(nn.Module):
    def __init__(self, input_nc=64, n_layers=2, use_sigmoid=True, gpu_ids=[]):
        super(VectorDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        
        ndf = 2*input_nc
        sequence = [
            nn.Linear(input_nc, ndf, bias=True),
            nn.LeakyReLU(0.1, inplace=False)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = 2**n
            sequence += [
                nn.Linear(ndf * nf_mult_prev, ndf * nf_mult, bias=True),
                nn.LeakyReLU(0.1, inplace=False)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = 2**(n_layers-1)
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = nf_mult//2
            sequence += [
                nn.Linear(ndf * nf_mult_prev, ndf * nf_mult, bias=True),
                nn.LeakyReLU(0.1, inplace=False)
            ]

        sequence += [nn.Linear(ndf * nf_mult, 1, bias=False)]
        
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)
        

class NoNormDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=True, gpu_ids=[]):
        super(NoNormDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)