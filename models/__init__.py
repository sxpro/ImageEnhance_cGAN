import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image enhancement
    if model == 'gan_singleD_ref':  # PSNR-oriented 
        from .unetgan_singleD_ref_model import RetouchModel as M
    # image restoration
    elif model == 'sr':  # PSNR-oriented super resolution
        from .SR_model import SRModel as M
    elif model == 'srgan':  # GAN-based super resolution, SRGAN / ESRGAN
        from .SRGAN_model import SRGANModel as M
    # video restoration
    elif model == 'video_base':
        from .Video_base_model import VideoBaseModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
