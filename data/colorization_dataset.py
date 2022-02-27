import os.path
from data.base_dataset import BaseDataset, get_transform
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2
from collections import  Counter
from tqdm import tqdm


class ColorizationDataset(BaseDataset):
    """This dataset class can load a set of natural images in RGB, and convert RGB format into (L, ab) pairs in Lab color space."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (L) and
        the nubmer of channels for output image is 2 (ab). The direction is from A to B
        """
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = [[self.opt.targetImage_path, self.opt.referenceImage_path]]
        self.ab_constant = np.load('./doc/ab_constant_filter.npy')
        self.transform_A = get_transform(self.opt, convert=False)
        self.transform_R = get_transform(self.opt, convert=False, must_crop=True)
        assert(opt.input_nc == 1 and opt.output_nc == 2)

    def __getitem__(self, index):
        path_A, path_R = self.AB_paths[index]
        im_A_l, im_A_ab, _ = self.process_img(path_A, self.transform_A)
        im_R_l, im_R_ab, hist = self.process_img(path_R, self.transform_R)

        im_dict = {
            'A_l': im_A_l,
            'A_ab': im_A_ab,
            'R_l': im_R_l,
            'R_ab': im_R_ab,
            'ab': self.ab_constant,
            'hist': hist,
            'A_paths': path_A
        }
        return im_dict


    def process_img(self, im_path, transform):

        weights_index = np.load('./doc/weight_index.npy')

        im = Image.open(im_path).convert('RGB')
        im = transform(im)
        im = self.__scale_width(im, 256)
        im = np.array(im)
        im = im[:16 * int(im.shape[0] / 16.0), :16 * int(im.shape[1] / 16.0), :]
        l_ts, ab_ts, gt_keys = [], [], []
        hist_total_new = np.zeros((441,), dtype=np.float32)
        for ratio in [0.25, 0.5, 1]:
            if ratio == 1:
                im_ratio = im
            else:
                im_ratio = cv2.resize(im, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
            lab = color.rgb2lab(im_ratio).astype(np.float32)

            if ratio == 1:
                ab_index_1 = np.round(lab[:, :, 1:] / 110.0 / 0.1) + 10.0
                keys_t = ab_index_1[:,:,0] * 21+ ab_index_1[:,:,1]
                keys_t_flatten = keys_t.flatten().astype(np.int32)
                dict_counter = dict(Counter(keys_t_flatten))
                for k, v in dict_counter.items():
                    hist_total_new[k] += v

                hist = hist_total_new[weights_index]
                hist = hist / np.sum(hist)

            lab_t = transforms.ToTensor()(lab)
            l_t = lab_t[[0], ...] / 50.0 - 1.0
            ab_t = lab_t[[1, 2], ...] / 110.0
            l_ts.append(l_t)
            ab_ts.append(ab_t)

        return l_ts, ab_ts, hist


    def __scale_width(self, img, target_width, method=Image.BICUBIC):
        ow, oh = img.size
        if ow <= oh:
            if (ow == target_width):
                return img
            w = target_width
            h = int(target_width * oh / ow)
        else:
            if (oh == target_width):
                return img
            h = target_width
            w = int(target_width * ow / oh)
        return img.resize((w, h), method)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
