##################################################
#borrowed from https://github.com/nashory/pggan-pytorch
##################################################
import torch
import numpy as np
import torchvision.models as models
import utils as utils
from tensorboardX import SummaryWriter
import os, sys


class TensorBoardX:
    def __init__(self,config_filename_list = [] , sub_dir ="" ):
        if sub_dir != "":
            sub_dir = '/' + sub_dir
        
        os.system('mkdir -p save{}'.format(sub_dir))
        for i in range(1000):
            self.path = 'save{}/try_{}'.format(sub_dir , i)
            if not os.path.exists(self.path):
                print("writing logs at {}".format(self.path))
                self.writer = {}
                self.writer['train'] = SummaryWriter(self.path+'/train')
                self.writer['val'] = SummaryWriter( self.path + '/val' )
                self.writer['test'] = SummaryWriter( self.path +'/test' )
                for config_filename in config_filename_list:
                    os.system('cp {} {}/'.format(config_filename , self.path))
                break
                
    def add_scalar(self, index, val, niter , logtype):
        self.writer[logtype].add_scalar(index, val, niter)

    def add_scalars(self, index, group_dict, niter , logtype):
        self.writer[logtype].add_scalar(index, group_dict, niter)

    def add_image_grid(self, index, ngrid, x, niter , logtype):
        grid = utils.make_image_grid(x, ngrid)
        self.writer[logtype].add_image(index, grid, niter)

    def add_image_single(self, index, x, niter , logtype):
        self.writer[logtype].add_image(index, x, niter)

    def add_graph(self, index, x_input, model , logtype):
        torch.onnx.export(model, x_input, os.path.join(self.path, "{}.proto".format(index)), verbose=True)
        self.writer[logtype].add_graph_onnx(os.path.join(self.path, "{}.proto".format(index)))

    def export_json(self, out_file , logtype ):
        self.writer[logtype].export_scalars_to_json(out_file)

