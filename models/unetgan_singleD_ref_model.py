import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import GANLoss, DiscLossWGANGP, L_cross, L_TV, L_spa, L_color
from models.archs.unet_arch import ContrastAdjustment
import utils.util as util
from metrics.niqe import calculate_niqe

logger = logging.getLogger('base')


class RetouchModel(BaseModel):
    def __init__(self, opt):
        super(RetouchModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.netC = networks.define_C(opt).to(self.device)
        # self.netD_pair = networks.define_D_pair(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            self.netC = DistributedDataParallel(self.netC, device_ids=[torch.cuda.current_device()])
            # self.netD_pair = DistributedDataParallel(self.netD_pair, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
            self.netC = DataParallel(self.netC)
            # self.netD_pair = DataParallel(self.netD_pair)

        if self.is_train:
            
            self.netD_H = networks.define_D(opt).to(self.device)
            if opt['dist']:
                self.netD_H = DistributedDataParallel(self.netD_H, device_ids=[torch.cuda.current_device()])
            else:
                self.netD_H = DataParallel(self.netD_H)
            self.netG.train()
            self.netC.train()
            self.netD_H.train()
            # self.netD_pair.train()
            # loss 
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'smoothl1':
                    self.cri_pix = nn.SmoothL1Loss(beta=0.01).to(self.device)
                elif l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            self.cri_l1 = nn.L1Loss().to(self.device)
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if train_opt['spa_weight'] > 0:
                self.cri_spa = L_spa(train_opt['spa_kernel']).to(self.device)
                self.l_spa_w = train_opt['spa_weight']
            else:
                self.cri_spa = None
                logger.info('Remove spa loss.')
            if train_opt['cross_weight'] > 0:
                self.cri_cross = L_cross().to(self.device)
                self.l_cross_w = train_opt['cross_weight']
            else:
                self.cri_cross = None
                logger.info('Remove cross loss.')
            if train_opt['color_weight'] > 0:
                l_color_type = train_opt['color_criterion']
                self.cri_color = L_color().to(self.device)
                self.l_color_w = train_opt['color_weight']
            else:
                self.cri_color = None
                logger.info('Remove color loss.')
            self.l_cond_dis_w = train_opt['cond_dis_weight']
            if self.l_cond_dis_w <= 0:
                logger.info('Remove cond_distance loss.')
            self.l_cond_code_w = train_opt['cond_code_weight']
            if self.l_cond_code_w <= 0:
                logger.info('Remove cond_code loss.')

            if train_opt['tv_weight'] > 0:
                self.cri_tv = L_TV(train_opt['tv_weight']).to(self.device)
            else:
                self.cri_tv = None
                logger.info('Remove tv loss.')
            if self.cri_fea:  # load VGG perceptual loss
                self.vgg_loss = vgg_loss(instance=True).to(self.device)
                if opt['dist']:
                    self.vgg_loss = DistributedDataParallel(self.vgg_loss, device_ids=[torch.cuda.current_device()])
                else:
                    self.vgg_loss = DataParallel(self.vgg_loss)

            # GD gan loss
            if self.opt['gan_type'] == 'wgan':
                self.criterionGAN = DiscLossWGANGP()
            else:
                self.criterionGAN = GANLoss(self.opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            self.l_gan_pair_w = train_opt['gan_pair_weight']
            # G_update_ratio and G_init_iters
            self.G_update_ratio = train_opt['G_update_ratio'] if train_opt['G_update_ratio'] else 1
            self.G_init_iters = train_opt['G_init_iters'] if train_opt['G_init_iters'] else 0
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # C
            wd_C = train_opt['weight_decay_C'] if train_opt['weight_decay_C'] else 0
            self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=train_opt['lr_C'],
                                                weight_decay=wd_C,
                                                betas=(train_opt['beta1_C'], train_opt['beta2_C']))
            self.optimizers.append(self.optimizer_C)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D_H = torch.optim.Adam(self.netD_H.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D_H)
            
            # self.optimizer_D_pair = torch.optim.Adam(self.netD_pair.parameters(), lr=train_opt['lr_D'],
                                                # weight_decay=wd_D,
                                                # betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            # self.optimizers.append(self.optimizer_D_pair)
            
            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        # print network
        self.print_network()
        self.load()

    def feed_data(self, data, GT=True, ref=True):
        self.real_H, self.ref, self.ref_alt = None, None, None
        
        self.var_L = data['LQ'].to(self.device)  # LQ
        if ref and 'ref' in  data:
            self.ref = data['ref'].to(self.device) # ref
            self.ref_path = data['ref_path']
        if ref and 'ref_alt' in  data:
            self.ref_alt = data['ref_alt'].to(self.device) # ref
            self.ref_path_alt = data['ref_path_alt']
        if GT:
            self.real_H = data['GT'].to(self.device)  # GT
        
    def backward_D_basic(self, netD, real, fake, ext=''):
        # Real
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake)
        if self.opt['gan_type'] == 'wgan':
            loss_D_real = torch.sigmoid(pred_real).mean()
            loss_D_fake = torch.sigmoid(pred_fake).mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD,
                                                real.data, fake.data)
            D_real = loss_D_real
            D_fake = loss_D_fake
        elif self.opt['gan_type'] == 'ragan':
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), 1.) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), 0.)) / 2
            D_real = torch.mean(torch.sigmoid(pred_real - torch.mean(pred_fake)))
            D_fake = torch.mean(torch.sigmoid(pred_fake - torch.mean(pred_real)))
        else:
            loss_D_real = self.criterionGAN(pred_real, 1.)
            loss_D_fake = self.criterionGAN(pred_fake, 0.)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            D_real = torch.mean(torch.sigmoid(pred_real))
            D_fake = torch.mean(torch.sigmoid(pred_fake))
        self.log_dict['D_real'+ext] = D_real.item()
        self.log_dict['D_fake'+ext] = D_fake.item()
        return loss_D

    def backward_G(self, step):
        # 
        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss between input out
                ###l_g_pix_ref_cycle = self.cri_pix(self.rec_ref_cycle, self.ref)
                l_g_pix_ref = self.cri_pix(self.rec_ref, self.ref)
                l_g_pix = self.l_pix_w *l_g_pix_ref
                ###l_g_pix = self.l_pix_w * 0.5*(l_g_pix_ref + l_g_pix_ref_cycle)
                l_g_total += l_g_pix
                self.log_dict['l_g_pix'] = l_g_pix.item()
                ## self.log_dict['l_g_pix_fake_ref'] = l_g_pix_fake_ref.item()
            if self.cri_cross:
                l_g_cross = self.l_cross_w * self.cri_cross(self.fake_H)
                l_g_total += l_g_cross
                self.log_dict['l_g_cross'] = l_g_cross.item()
            if self.cri_color:
                l_g_color = self.l_color_w * self.cri_color(self.fake_H, self.var_L)
                l_g_total += l_g_color
                self.log_dict['l_g_color'] = l_g_color.item()
            if self.cri_tv:
                l_g_tv = self.cri_tv(self.fake_H)
                l_g_total += l_g_tv
                self.log_dict['l_g_tv'] = l_g_tv.item()
            if self.cri_fea:  # feature loss between input out
                l_g_fea = self.l_fea_w * self.vgg_loss(vgg_process(self.fake_H), vgg_process(self.var_L))
                l_g_total += l_g_fea
                self.log_dict['l_g_fea'] = l_g_fea.item()
            if self.cri_spa:
                l_spa = self.l_spa_w * torch.mean(self.cri_spa(self.fake_ref, self.ref))
                l_g_total += l_spa
                self.log_dict['l_spa'] = l_spa.item()
            if self.cri_color:
                pass
            # loss on cond distribution
            if self.l_cond_dis_w >0 :
                l_g_cond_dis_fakeH_ref = self.l_cond_dis_w * nn.functional.mse_loss(self.cond_fake_H, self.cond_ref)
                l_g_total += l_g_cond_dis_fakeH_ref
                self.log_dict['l_g_cond_fakeH_dis'] = l_g_cond_dis_fakeH_ref.item()
            if self.l_cond_code_w >0 :
                diff_fakeH_refH = (self.cond_fake_H - self.cond_ref).squeeze()
                diff_fakeH_L = (self.cond_fake_H  - self.cond_L).squeeze()
                diff_refH_L = (self.cond_ref - self.cond_L).squeeze()
                self.tau = torch.mean(diff_refH_L*diff_refH_L, dim=-1)+1e-10
                self.positive = torch.mean(diff_fakeH_refH*diff_fakeH_refH, dim=-1)
                self.negative = torch.mean(diff_fakeH_L*diff_fakeH_L, dim=-1)
                positive = self.positive/self.tau
                negative = self.negative/self.tau
                l_g_cond_fakeH = self.l_cond_code_w * nn.functional.l1_loss(positive / (positive + negative), torch.zeros_like(self.positive))
                l_g_total += l_g_cond_fakeH
                self.log_dict['l_g_cond_code'] = l_g_cond_fakeH.item()
            if self.opt['use_gan']:
                #gan G loss
                # H
                pred_g_fake = self.netD_H(self.netC(self.fake_H))
                pred_g_fake_ref = self.netD_H(self.netC(self.fake_ref))
                if self.opt['gan_type'] == 'gan':
                    l_g_gan = self.l_gan_w * ( 0.9*self.criterionGAN(pred_g_fake, 1.) + 0.1*self.criterionGAN(pred_g_fake_ref, 1.) )
                elif self.opt['gan_type'] == 'wgan':
                    l_g_gan = -(0.9*pred_g_fake.mean() + 0.1*pred_g_fake_ref.mean()) * self.l_gan_w
                elif self.opt['gan_type'] == 'ragan':
                    pred_d_real = self.netD_H(self.cond_ref).detach()
                    l_g_gan = self.l_gan_w * 0.5* (
                        (self.criterionGAN(pred_d_real - torch.mean(pred_g_fake), 0.) +
                        self.criterionGAN(pred_g_fake - torch.mean(pred_d_real), 1.)) * 0.9 + (self.criterionGAN(pred_d_real - torch.mean(pred_g_fake_ref), 0.) +
                        self.criterionGAN(pred_g_fake_ref - torch.mean(pred_d_real), 1.)) * 0.1
                        )
                l_g_total += l_g_gan
                self.log_dict['l_g_gan'] = l_g_gan.item()
            
            self.l_g_total = l_g_total
            self.log_dict['l_g_total'] = l_g_total.item()
            return l_g_total

    def augmentation(self, input): # randomly augment some input
        aug_seed = torch.rand(1)
        if aug_seed<0.001:  
            # adaptive Gaussian Noise
            bg_noise_std = 0.3 * (0.2+0.8*torch.rand(1).item()) * torch.std(input, dim=[1,2,3], keepdim=True)
            ada_noise_std = 0.04 * input.clamp(max=0.5)
            input_aug = (input + bg_noise_std*torch.randn_like(input) + ada_noise_std*torch.randn_like(input)).clamp_(min=0., max=1.)
        elif aug_seed < 0.001:
            # quantization error
            stairs = 64
            input_aug = torch.floor(input*stairs)/stairs
        else:
            input_aug = input
            
        return input_aug


    def light_encode(self, input): # -> (batchsize, cond_nf)
        return torch.mean(self.netC(input), dim=[2,3], keepdim=False)


    def forward(self, step):
        self.fake_H = self.netG(self.netC, self.var_L, self.ref)
        
        ref_aug = self.augmentation(self.ref)
        self.fake_ref = self.netG(self.netC, ref_aug, self.ref_alt)
        self.rec_ref = self.netG(self.netC, ref_aug, self.ref)
        
        
    def optimize_parameters(self, step):
        #netC
        self.optimizer_C.zero_grad()
        #turn off D gradients
        for p in self.netD_H.parameters():
            p.requires_grad = False
        #netG
        self.optimizer_G.zero_grad()
        self.forward(step)
        l_g_total = self.backward_G(step)
        if l_g_total:
            l_g_total.backward()
        # update netG
        self.optimizer_G.step()
        
        if self.opt['use_gan']:
            #turn on D gradients
            for p in self.netD_H.parameters():
                p.requires_grad = True
            self.optimizer_D_H.zero_grad()
            if step % self.G_update_ratio == 0 and step > self.G_init_iters:
                cond_ref = self.netC(self.ref)
                cond_fake_H_detach = self.netC(self.fake_H.detach())
                l_d = self.l_gan_w * self.backward_D_basic(self.netD_H, cond_ref, cond_fake_H_detach, ext='')
                self.log_dict['l_d'] = l_d.item()
                l_d.backward()
                # update netD_H 
                self.optimizer_D_H.step()
        
        # update netC
        self.optimizer_C.step() 


    def test(self, ref_cri):
        self.netG.eval()
        self.netC.eval()
        # self.netD_pair.eval()
        with torch.no_grad():
        
            index = 0
            if ref_cri == 'mse_GT': # if provided with GT
                mse = 1e6 # choose smallest
                for i in range(self.ref.size()[1]):
                    fake_H_i = self.netG(self.netC, self.var_L, self.ref[:,i,:,:,:])
                    mse_i = torch.pow(self.real_H - fake_H_i,2).sum(dim=[1,2,3])
                    if mse > mse_i: # choose smallest
                        mse = mse_i
                        index = i
            elif ref_cri == 'color_condition': # compare the condition codes
                cond_nf = self.opt['condition_nf']
                cond_L = self.light_encode(self.var_L)
                illum_L = torch.mean(self.var_L, dim=[1,2,3], keepdim=True)
                cond_diff = 1e6  # choose smallest
                for i in range(self.ref.size()[1]):
                    illum_ref_i = torch.mean(self.ref[:,i,:,:,:], dim=[1,2,3], keepdim=True)
                    if illum_ref_i < 0.3:
                        continue
                    cond_ref_i = self.light_encode(self.ref[:,i,:,:,:]/illum_ref_i*illum_L)
                    cond_diff_i = nn.functional.mse_loss(cond_L, cond_ref_i)
                    if cond_diff > cond_diff_i:
                        cond_diff = cond_diff_i
                        index = i
            elif ref_cri == 'niqe': # if no GT is provided for validation
                niqe = 1e6 # choose smallest
                for i in range(self.ref.size()[1]):
                    fake_H_i = self.netG(self.netC, self.var_L, self.ref[:,i,:,:,:])
                    fake_H_i_img = util.tensor2img(fake_H_i[0,:,:,:])  # uint8
                    try:
                        niqe_i = calculate_niqe(fake_H_i_img, 0).flatten()[0]
                        if niqe > niqe_i: # choose smallest
                            niqe = niqe_i
                            index = i
                    except:
                        niqe_i = None
                        logger.info('NIQE calculation failed')

            elif ref_cri == 'random':
                index = torch.randint(low=0, high=self.ref.size()[1], size=(1,))[0]
            else: 
                raise NotImplementedError('Selection criteria ref_cri: [{:s}] not recognized. Please choose from mse_GT/niqe/random.'.format(ref_cri))
        
            self.ref = self.ref[:,index,:,:,:]
            
            if self.is_train:
                ref_aug = self.augmentation(self.ref)
                self.fake_H = self.netG(self.netC, self.var_L, self.ref)
                self.rec_ref = self.netG(self.netC, ref_aug, self.ref)
                if self.ref_alt is not None:
                    self.fake_ref = self.netG(self.netC, ref_aug, self.ref_alt)
                    self.rec_ref_cycle = self.netG(self.netC, self.fake_ref, self.ref)
                else: 
                    self.fake_ref = None
                    self.rec_ref_cycle = None
            else:
                self.fake_H = self.netG(self.netC, self.var_L, self.ref)
                if self.ref_alt is not None:
                    self.fake_ref = self.netG(self.netC, ref_aug, self.ref_alt)
                    self.rec_ref_cycle = self.netG(self.netC, self.fake_ref, self.ref)
                else: 
                    self.fake_ref = None
                    self.rec_ref_cycle = None
            
        self.netG.train()
        self.netC.train()
        # self.netD_pair.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if self.is_train:
            out_dict['rec_ref'] = self.rec_ref.detach()[0].float().cpu()
            if self.ref_alt is not None:
                out_dict['fake_ref'] = self.fake_ref.detach()[0].float().cpu()
                out_dict['rec_ref_cycle'] = self.rec_ref_cycle.detach()[0].float().cpu()
        out_dict['ref'] = self.ref.detach()[0].float().cpu()

        if GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        # Condition Encoder
        s, n = self.get_network_description(self.netC)
        if isinstance(self.netC, nn.DataParallel) or isinstance(self.netC, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netC.__class__.__name__,
                                             self.netC.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netC.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network C structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            for netD in [self.netD_H]:
                s, n = self.get_network_description(netD)
                if isinstance(netD, nn.DataParallel) or isinstance(netD,
                                                                        DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(netD.__class__.__name__,
                                                    netD.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(netD.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_C = self.opt['path']['pretrain_model_C']
        if load_path_C is not None:
            logger.info('Loading model for C [{:s}] ...'.format(load_path_C))
            self.load_network(load_path_C, self.netC, self.opt['path']['strict_load'])
        load_path_D_H = self.opt['path']['pretrain_model_D_H']
        if load_path_D_H is not None:
            logger.info('Loading model for D_H [{:s}] ...'.format(load_path_D_H))
            self.load_network(load_path_D_H, self.netD_H, self.opt['path']['strict_load'])
        load_path_D_pair = self.opt['path']['pretrain_model_D_pair']

    def updateG(self, new_model_dict):
        if isinstance(self.netG, nn.DataParallel):
            network = self.netG.module
            network.load_state_dict(new_model_dict)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.netC, 'C', iter_label)
        self.save_network(self.netD_H, 'D_H', iter_label)
