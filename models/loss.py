import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class DiscLossWGANGP():
    def __init__(self):
        self.LAMBDA = 10

    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self, opt, tensor):
        # DiscLossLS.initialize(self, opt, tensor)
        self.LAMBDA = 10

    # def get_g_loss(self, net, realA, fakeB):
    #     # First, G(A) should fake the discriminator
    #     self.D_fake = net.forward(fakeB)
    #     return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1, eps=1e-6):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.eps=eps

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (h_x-1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class vgg_loss(nn.Module):
    def __init__(self, instance=False):
        super(vgg_loss, self).__init__()
        vgg_model = models.vgg19(pretrained=True)
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }
        self.instance = instance
    def _forward(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output
    def forward(self, img, gt):
        mse = nn.MSELoss(size_average=True)
        img_vgg = self._forward(img)
        gt_vgg = self._forward(gt)
        if self.instance:
            return mse(F.instance_norm(img_vgg[0]), F.instance_norm(gt_vgg[0])) + 0.6 * mse(F.instance_norm(img_vgg[1]), F.instance_norm(gt_vgg[1])) + \
                   0.4 * mse(F.instance_norm(img_vgg[2]),F.instance_norm(gt_vgg[2])) + 0.2 * mse(F.instance_norm(img_vgg[3]), F.instance_norm(gt_vgg[3]))
        else:
            return mse(img_vgg[0], gt_vgg[0]) + 0.6 * mse(img_vgg[1], gt_vgg[1]) + \
                   0.4 * mse(img_vgg[2],gt_vgg[2]) + 0.2 * mse(img_vgg[3], gt_vgg[3])


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()
        self.cri = nn.SmoothL1Loss(beta=0.001)

    def forward(self, x, y):
        b, c, h, w = x.shape
        x_norm = F.layer_norm(x, x.size()[1:], eps=1e-4)
        y_norm = F.layer_norm(y, y.size()[1:], eps=1e-4)
        mean_rgb_x = torch.mean(x_norm, [2, 3], keepdim=False)
        mean_rgb_y = torch.mean(y_norm, [2, 3], keepdim=False)
        
        mean_rgb_diff = mean_rgb_x - mean_rgb_y
        mr, mg, mb = torch.split(mean_rgb_diff, 1, dim=1)
        
        zero_tensor = torch.zeros_like(mr)
        
        Drg = self.cri(mr - mg, zero_tensor)
        Drb = self.cri(mr - mb, zero_tensor)
        Dgb = self.cri(mb - mg, zero_tensor)
        k = Drg + Drb + Dgb
        # Drg = torch.pow(mr - mg, 2)
        # Drb = torch.pow(mr - mb, 2)
        # Dgb = torch.pow(mb - mg, 2)
        # k = torch.pow(Drg + Drb + Dgb + 1e-10, 0.5)
        return k


class L_cross(nn.Module):
    def __init__(self):
        super(L_cross, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(3,stride=1)
        self.pad_pool = nn.ReflectionPad2d(1)
        self.pad_kernel = nn.ReflectionPad2d(1)
        self.cri = nn.SmoothL1Loss(beta=0.01)

    def forward(self, enhance):
        enhance_base = enhance + 1e-2 # no pool 
        enhance_padded = self.pad_kernel(enhance)
        
        D_enhance_letf = F.conv2d(enhance_padded, self.weight_left, padding=0, groups=3)
        D_enhance_right = F.conv2d(enhance_padded, self.weight_right, padding=0, groups=3)
        D_enhance_up = F.conv2d(enhance_padded, self.weight_up, padding=0, groups=3)
        D_enhance_down = F.conv2d(enhance_padded, self.weight_down, padding=0, groups=3)
        
        ## l1
        zero_tensor = torch.zeros_like(enhance_base[:,0,:,:])
        Drg_left = self.cri(D_enhance_letf[:,0,:,:]*enhance_base[:,1,:,:] - D_enhance_letf[:,1,:,:]*enhance_base[:,0,:,:], zero_tensor)
        Drg_right = self.cri(D_enhance_right[:,0,:,:]*enhance_base[:,1,:,:] - D_enhance_right[:,1,:,:]*enhance_base[:,0,:,:], zero_tensor)
        Drg_up = self.cri(D_enhance_up[:,0,:,:]*enhance_base[:,1,:,:] - D_enhance_up[:,1,:,:]*enhance_base[:,0,:,:], zero_tensor)
        Drg_down = self.cri(D_enhance_down[:,0,:,:]*enhance_base[:,1,:,:] - D_enhance_down[:,1,:,:]*enhance_base[:,0,:,:], zero_tensor)
        Drg = Drg_left + Drg_right + Drg_up + Drg_down
        Drb_left = self.cri(D_enhance_letf[:,0,:,:]*enhance_base[:,2,:,:] - D_enhance_letf[:,2,:,:]*enhance_base[:,0,:,:], zero_tensor)
        Drb_right = self.cri(D_enhance_right[:,0,:,:]*enhance_base[:,2,:,:] - D_enhance_right[:,2,:,:]*enhance_base[:,0,:,:], zero_tensor)
        Drb_up = self.cri(D_enhance_up[:,0,:,:]*enhance_base[:,2,:,:] - D_enhance_up[:,2,:,:]*enhance_base[:,0,:,:], zero_tensor)
        Drb_down = self.cri(D_enhance_down[:,0,:,:]*enhance_base[:,2,:,:] - D_enhance_down[:,2,:,:]*enhance_base[:,0,:,:], zero_tensor)
        Drb = Drb_left + Drb_right + Drb_up + Drb_down
        Dgb_left = self.cri(D_enhance_letf[:,1,:,:]*enhance_base[:,2,:,:] - D_enhance_letf[:,2,:,:]*enhance_base[:,1,:,:], zero_tensor)
        Dgb_right = self.cri(D_enhance_right[:,1,:,:]*enhance_base[:,2,:,:] - D_enhance_right[:,2,:,:]*enhance_base[:,1,:,:], zero_tensor)
        Dgb_up = self.cri(D_enhance_up[:,1,:,:]*enhance_base[:,2,:,:] - D_enhance_up[:,2,:,:]*enhance_base[:,1,:,:], zero_tensor)
        Dgb_down = self.cri(D_enhance_down[:,1,:,:]*enhance_base[:,2,:,:] - D_enhance_down[:,2,:,:]*enhance_base[:,1,:,:], zero_tensor)
        Dgb = Dgb_left + Dgb_right + Dgb_up + Dgb_down
        ## l2
        # Drg_left = torch.pow(D_enhance_letf[:,0,:,:]*enhance_base[:,1,:,:] - D_enhance_letf[:,1,:,:]*enhance_base[:,0,:,:], 2).mean(dim=[0,1,2])
        # Drg_right = torch.pow(D_enhance_right[:,0,:,:]*enhance_base[:,1,:,:] - D_enhance_right[:,1,:,:]*enhance_base[:,0,:,:], 2).mean(dim=[0,1,2])
        # Drg_up = torch.pow(D_enhance_up[:,0,:,:]*enhance_base[:,1,:,:] - D_enhance_up[:,1,:,:]*enhance_base[:,0,:,:], 2).mean(dim=[0,1,2])
        # Drg_down = torch.pow(D_enhance_down[:,0,:,:]*enhance_base[:,1,:,:] - D_enhance_down[:,1,:,:]*enhance_base[:,0,:,:], 2).mean(dim=[0,1,2])
        # Drg = Drg_left + Drg_right + Drg_up + Drg_down
        # Drb_left = torch.pow(D_enhance_letf[:,0,:,:]*enhance_base[:,2,:,:] - D_enhance_letf[:,2,:,:]*enhance_base[:,0,:,:], 2).mean(dim=[0,1,2])
        # Drb_right = torch.pow(D_enhance_right[:,0,:,:]*enhance_base[:,2,:,:] - D_enhance_right[:,2,:,:]*enhance_base[:,0,:,:], 2).mean(dim=[0,1,2])
        # Drb_up = torch.pow(D_enhance_up[:,0,:,:]*enhance_base[:,2,:,:] - D_enhance_up[:,2,:,:]*enhance_base[:,0,:,:], 2).mean(dim=[0,1,2])
        # Drb_down = torch.pow(D_enhance_down[:,0,:,:]*enhance_base[:,2,:,:] - D_enhance_down[:,2,:,:]*enhance_base[:,0,:,:], 2).mean(dim=[0,1,2])
        # Drb = Drb_left + Drb_right + Drb_up + Drb_down
        # Dgb_left = torch.pow(D_enhance_letf[:,1,:,:]*enhance_base[:,2,:,:] - D_enhance_letf[:,2,:,:]*enhance_base[:,1,:,:], 2).mean(dim=[0,1,2])
        # Dgb_right = torch.pow(D_enhance_right[:,1,:,:]*enhance_base[:,2,:,:] - D_enhance_right[:,2,:,:]*enhance_base[:,1,:,:], 2).mean(dim=[0,1,2])
        # Dgb_up = torch.pow(D_enhance_up[:,1,:,:]*enhance_base[:,2,:,:] - D_enhance_up[:,2,:,:]*enhance_base[:,1,:,:], 2).mean(dim=[0,1,2])
        # Dgb_down = torch.pow(D_enhance_down[:,1,:,:]*enhance_base[:,2,:,:] - D_enhance_down[:,2,:,:]*enhance_base[:,1,:,:], 2).mean(dim=[0,1,2])
        # Dgb = Dgb_left + Dgb_right + Dgb_up + Dgb_down
        E = Drg + Drb + Dgb

        return E

def gaussian_kernel(kernel_size = 7, sig = 1.0):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    center = kernel_size//2, 
    x_axis = np.linspace(0, kernel_size-1, kernel_size) - center
    y_axis = np.linspace(0, kernel_size-1, kernel_size) - center
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig)) / (np.sqrt(2*np.pi)*sig)
    return kernel
    

class L_spa(nn.Module):
    def __init__(self, pool_kernel_size):
        super(L_spa, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        # self.pool = nn.AvgPool2d(pool_kernel_size,stride=1)
        self.gaussian = torch.FloatTensor(gaussian_kernel(pool_kernel_size, pool_kernel_size/4.0)).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)

    def forward(self, enhance, org):
        b, c, h, w = org.shape
        
        # # ### gray version
        # # org_mean = torch.mean(org, 1, keepdim=True)
        # # enhance_mean = torch.mean(enhance, 1, keepdim=True)
        # ### color version
        org_pool = F.conv2d(org, self.gaussian, padding=0, groups=3)
        enhance_pool = F.conv2d(enhance, self.gaussian, padding=0, groups=3)
        
        # weight_diff = torch.max(
        #     torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
        #                                                       torch.FloatTensor([0]).cuda()),
        #     torch.FloatTensor([0.5]).cuda())
        # E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)
        
        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=0, groups=3)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=0, groups=3)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=0, groups=3)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=0, groups=3)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=0, groups=3)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=0, groups=3)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=0, groups=3)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=0, groups=3)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = D_left + D_right + D_up + D_down

        return E


class L_exp(nn.Module):

    def __init__(self, patch_size=16, mean_val=0.05):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        # y = torch.mean(y, 1, keepdim=True)
        # mean_y = self.pool(y)
        # d = torch.mean(torch.pow(mean_y-mean, 2))
        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d