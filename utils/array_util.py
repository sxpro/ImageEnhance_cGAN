import torch
import torch.nn as nn


def pad_tensor(input, divide):
    height_org, width_org = input.shape[2], input.shape[3]

    if width_org % divide != 0 or height_org % divide != 0:
        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]
    
"""
Convert an RGB tensor to gray, [N, 3, H, W] 
"""
def rgb2gray(input):
    out = 0.2125*input[:,0:1,:,:] + 0.7154*input[:,1:2,:,:] + 0.0721*input[:,2:3,:,:]
    return out.expand(-1,3,-1,-1)
    