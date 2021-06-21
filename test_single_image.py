import os
import sys
import cv2

import os.path as osp
from tqdm import tqdm
from glob import glob

import logging
import time
import argparse
from collections import OrderedDict
import numpy as np
import torch
import options.options as option
from models import create_model


sys.path.insert(0, "../../")
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader

#### options
parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt", type=str, default="options/test/test_Enhance_gan_ref.yml", help="Path to options YMAL file."
)
parser.add_argument("-input_dir", type=str, default="low/")
parser.add_argument("-ref_dir", type=str, default="ref/")
parser.add_argument("-output_dir", type=str, default="output/")
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model = create_model(opt)

if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)

test_files = glob(osp.join(args.input_dir, "*png"))
ref_files = glob(osp.join(args.ref_dir, "*png"))
for inx, path in tqdm(enumerate(test_files)):
    name = path.split("/")[-1].split(".")[0]

    img = cv2.imread(path)[:, :, [2, 1, 0]]
    img = img.transpose(2, 0, 1)[None] / 255
    img_t = torch.as_tensor(np.ascontiguousarray(img)).float()
    for ref_inx, ref_path in tqdm(enumerate(ref_files)):
        data = dict()
        ref_name = ref_path.split("/")[-1].split(".")[0]
        ref_img = cv2.imread(ref_path)[:, :, [2, 1, 0]]
        ref_img = ref_img.transpose(2, 0, 1)[None] / 255
        ref_img_t = torch.as_tensor(np.ascontiguousarray(ref_img)).float().unsqueeze(1)
        data['LQ'] = img_t
        data['ref'] = ref_img_t
        data['ref_path'] = ref_path
        data['GT'] = img_t
        model.feed_data(data)
        model.test('random')

        out = model.fake_H.detach().float().cpu()[0]
        out_im = util.tensor2img(out)

        save_path = osp.join(args.output_dir, "{}_ref_{}.png".format(name, ref_name))
        cv2.imwrite(save_path, out_im)
