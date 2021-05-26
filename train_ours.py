import os
import math
import argparse
import numpy as np
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from metrics.niqe import calculate_niqe


def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    torch.cuda.empty_cache()
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default="options/train/train_Enhance_gan_ref.yml", help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = np.random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        np.random.seed() # reset seed
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data, GT=False, ref=opt['ref'])
            model.optimize_parameters(current_step)
            
            # # check random reference selection 
            # LQ_path = os.path.normpath(train_data['LQ_path'][0]).split(os.sep)
            # ref_path = os.path.normpath(train_data['ref_path'][0]).split(os.sep)
            # print(f"{LQ_path[-1]} -- {ref_path[-1]}")
            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                if opt['model'] in ['inn', 'unet', 'ganref', 'gan_singleD_ref', 'sr'] and rank <= 0:  # image restoration validation
                    with_GT = val_loader.dataset.opt['dataroot_GT']
                    ref_cri = val_loader.dataset.opt['ref_cri']
                    if ref_cri == 'mse_GT':
                        if not with_GT:
                            ref_cri = 'color_condition'
                            logger.info('No GT provided, use "color_condition", "random" or "niqe" as the reference image selection criteria. Here we replace with "color_condition".')
                    
                    # does not support multi-GPU validation
                    pbar = util.ProgressBar(len(val_loader))
                    avg_niqe, avg_psnr_ref = 0., 0.
                    if with_GT:
                        avg_psnr, avg_ssim = 0., 0.
                        
                    idx = 0
                    for val_data in val_loader:
                        idx += 1
                        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                   
                        img_dir = os.path.join(opt['path']['val_images'], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(val_data, GT=with_GT, ref=opt['ref'])
                        model.test(ref_cri=ref_cri)

                        visuals = model.get_current_visuals()
                        sr_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8
                        ref_img = util.tensor2img(visuals['ref'])
                        rec_ref_img = util.tensor2img(visuals['rec_ref'])
                        
                        
                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir,
                                                     '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)

                        # calculate NIQE
                        try:
                            avg_niqe +=calculate_niqe(sr_img, 0).flatten()[0]
                        except:
                            logger.info('NIQE calculation failed')
                        # calculate Reconstruction PSNR
                        avg_psnr_ref += util.calculate_psnr(rec_ref_img, ref_img)
                        pbar.update('Test {}'.format(img_name))
                        if with_GT:
                            # calculate PSNR
                            avg_psnr += util.calculate_psnr(sr_img, gt_img)
                            avg_ssim += util.calculate_ssim(sr_img, gt_img)
                        if 'L_ref' in visuals:
                            fake_lr_img = util.tensor2img(visuals['L_ref'])  # uint8
                            save_lr_img_path = os.path.join(img_dir,
                                                         'L_ref_{:s}_{:d}.png'.format(img_name, current_step))
                            util.save_img(fake_lr_img, save_lr_img_path)
                        if 'rec_ref' in visuals:
                            fake_lr_img = util.tensor2img(visuals['rec_ref'])  # uint8
                            save_lr_img_path = os.path.join(img_dir,
                                                         'rec_ref_{:s}_{:d}.png'.format(img_name, current_step))
                            util.save_img(fake_lr_img, save_lr_img_path)
                        if 'fake_ref' in visuals:
                            fake_lr_img = util.tensor2img(visuals['fake_ref'])  # uint8
                            save_lr_img_path = os.path.join(img_dir,
                                                         'fake_ref_{:s}_{:d}.png'.format(img_name, current_step))
                            util.save_img(fake_lr_img, save_lr_img_path)
                        if 'rec_L' in visuals:
                            fake_lr_img = util.tensor2img(visuals['rec_L'])  # uint8
                            save_lr_img_path = os.path.join(img_dir,
                                                         'rec_L_{:s}_{:d}.png'.format(img_name, current_step))
                            util.save_img(fake_lr_img, save_lr_img_path)
                    
                    avg_niqe = avg_niqe / idx
                    avg_psnr_ref = avg_psnr_ref / idx
                    if with_GT:
                        avg_psnr = avg_psnr / idx
                        avg_ssim = avg_ssim / idx
                        # log
                        logger.info('# Validation # PSNR: {:.4e}, # SSIM: {:.4e}, # Ref PSNR: {:.4e}; NIQE: {:.4e}'.format(avg_psnr, avg_ssim, avg_psnr_ref, avg_niqe))
                    else:
                        # log
                        logger.info('# Validation # Ref PSNR: {:.4e}; NIQE: {:.4e}'.format(avg_psnr, avg_ssim, avg_psnr_rec, avg_niqe))
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)
                else:  
                    raise KeyError('Model not supported, got {}.'.format(opt['model']))

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        tb_logger.close()


if __name__ == '__main__':
    main()
