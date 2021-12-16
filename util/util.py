"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from scipy import linalg
from PIL import Image
from skimage import color
import os


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def calc_hist(data_ab, device):
    N, C, H, W = data_ab.shape
    grid_a = torch.linspace(-1, 1, 21).view(1, 21, 1, 1, 1).expand(N, 21, 21, H, W).to(device)
    grid_b = torch.linspace(-1, 1, 21).view(1, 1, 21, 1, 1).expand(N, 21, 21, H, W).to(device)
    hist_a = torch.max(0.1 - torch.abs(grid_a - data_ab[:, 0, :, :].view(N, 1, 1, H, W)), torch.Tensor([0]).to(device)) * 10
    hist_b = torch.max(0.1 - torch.abs(grid_b - data_ab[:, 1, :, :].view(N, 1, 1, H, W)), torch.Tensor([0]).to(device)) * 10
    hist = (hist_a * hist_b).mean(dim=(3, 4)).view(N, -1)
    return hist


class Convert(object):
    def __init__(self, device):
        xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                                 [0.212671, 0.715160, 0.072169],
                                 [0.019334, 0.119193, 0.950227]])
        rgb_from_xyz = linalg.inv(xyz_from_rgb)
        self.rgb_from_xyz = torch.Tensor(rgb_from_xyz).to(device)
        self.channel_mask = torch.Tensor([1, 0, 0]).to(device)
        self.xyz_weight = torch.Tensor([0.95047, 1., 1.08883]).to(device)
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
        self.zero = torch.Tensor([0]).to(device)
        self.one = torch.Tensor([1]).to(device)

    def lab2rgb(self, img):
        img = img.permute(0, 2, 3, 1)
        img1 = (img + 1.0) * 50.0 * self.channel_mask
        img2 = img * 110.0 * (1 - self.channel_mask)
        img = img1 + img2
        return self.xyz2rgb(self.lab2xyz(img))

    def lab2xyz(self, img):
        L, a, b = img[:, :, :, 0], img[:, :, :, 1], img[:, :, :, 2]
        y = (L + 16.) / 116.
        x = (a / 500.) + y
        z = y - (b / 200.)
        z = torch.max(z, self.zero)
        out = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)
        mask = (out > 0.2068966).float()
        out1 = torch.pow(out, 3) * mask
        out2 = (out - 16.0 / 116.) / 7.787 * (1 - mask)
        out = out1 + out2
        out *= self.xyz_weight
        return out

    def xyz2rgb(self, img):
        arr = img.matmul(self.rgb_from_xyz.t())
        mask = (arr > 0.0031308).float()
        arr1 = (1.055 * torch.pow(torch.max(arr, self.zero), 1 / 2.4) - 0.055) * mask
        arr2 = arr * 12.92 * (1 - mask)
        arr = arr1 + arr2
        arr = torch.min(torch.max(arr, self.zero), self.one)
        return arr

    def rgb_norm(self, img):
        img = (img - self.mean) / self.std
        return img.permute(0, 3, 1, 2)