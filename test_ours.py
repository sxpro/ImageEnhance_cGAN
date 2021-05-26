import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import torch
import numpy as np

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model
from metrics.niqe import calculate_niqe

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='options/test/test_Enhance_gan_ref.yml', help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
#torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    #### random seed
    seed = test_loader.dataset.opt['manual_seed']
    if seed is None:
        seed = np.random.randint(1, 1000)
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)
    
    with_GT = test_loader.dataset.opt['dataroot_GT']
    ref_cri = test_loader.dataset.opt['ref_cri']
    logger.info('Selection criteria for the reference image is: [{:s}].'.format(ref_cri))
    if ref_cri == 'mse_GT':
        if not with_GT:
            ref_cri = 'niqe'
            logger.info('No GT provided, use "color_condition", "random" or "niqe" as the reference image selection criteria. Here we replace with "niqe".')
    
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['niqe'] = []
    
    for data in test_loader:
        model.feed_data(data, GT=with_GT, ref=opt['ref'])

        img_path = data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test(ref_cri=ref_cri)
        visuals = model.get_current_visuals(GT=with_GT)

        sr_img = util.tensor2img(visuals['rlt'])  # uint8

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)
        
        # calculate NIQE
        try:
            niqe = calculate_niqe(sr_img, 0).flatten()[0]
            test_results['niqe'].append(niqe)
        except:
            niqe = 0
            logger.info('NIQE calculation failed')
        # calculate PSNR and SSIM
        if with_GT:
            gt_img = util.tensor2img(visuals['GT'])
            psnr = util.calculate_psnr(sr_img, gt_img)
            ssim = util.calculate_ssim(sr_img, gt_img)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}  NIQE: {:.6f};.'.format(img_name, psnr, ssim, niqe))
        else:
            if niqe:
                logger.info('{:20s} - NIQE: {:.6f}.'.format(img_name, niqe))
    
    avg_niqe = sum(test_results['niqe']) / len(test_results['niqe'])
    # metrics
    if with_GT: 
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info(
            '----Average PSNR/SSIM/NIQE results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}  NIQE: {:.6f};\n'.format(
                test_set_name, ave_psnr, ave_ssim, avg_niqe))
    else:
        logger.info(
            '----Average NIQE results for {}----\n\tNIQE: {:.6f}\n'.format(test_set_name, avg_niqe))
