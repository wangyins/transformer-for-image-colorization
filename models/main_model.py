from .base_model import BaseModel
from . import networks
from util import util


class MainModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', dataset_mode='aligned')
        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.bias_input_nc, opt.output_nc, opt.norm, opt.init_type,
                                      opt.init_gain, self.gpu_ids)
        self.convert = util.Convert(self.device)

    def set_input(self, input):
        self.image_paths = input['A_paths']
        self.ab_constant = input['ab'].to(self.device)
        self.hist = input['hist'].to(self.device)

        self.real_A_l, self.real_A_ab, self.real_R_l, self.real_R_ab, self.real_R_histogram = [], [], [], [], []
        for i in range(3):
            self.real_A_l += input['A_l'][i].to(self.device).unsqueeze(0)
            self.real_A_ab += input['A_ab'][i].to(self.device).unsqueeze(0)
            self.real_R_l += input['R_l'][i].to(self.device).unsqueeze(0)
            self.real_R_ab += input['R_ab'][i].to(self.device).unsqueeze(0)
            self.real_R_histogram += [util.calc_hist(input['A_ab'][i].to(self.device), self.device)]

    def forward(self):
        self.fake_imgs = self.netG(self.real_A_l[-1], self.real_R_l[-1], self.real_R_ab[0], self.hist,
                                   self.ab_constant, self.device)
        self.fake_R_histogram = []
        for i in range(3):
            self.fake_R_histogram += [util.calc_hist(self.fake_imgs[i], self.device)]
