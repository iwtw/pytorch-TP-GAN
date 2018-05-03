#wrappers for convenience
import torch.nn as nn
from torch.nn.init import xavier_normal , kaiming_normal
import copy

def weight_initialization( weight , activation ):
    if  type(activation) == type(nn.LeakyReLU()):
        kaiming_normal( weight , a = activation.negative_slope )
    elif type(activation) == type(nn.ReLU())  or type(activation) == type(nn.PReLU()) :
        kaiming_normal( weight , a = 0 )
    else:
        xavier_normal( weight )
    return

def conv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 , activation= None , use_batchnorm = False , preactivation = False , bias = True):
    if use_batchnorm:
        bias = False

    layers = []
    if preactivation and activation is not None:
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        layers.append( activation )
    conv = nn.Conv2d( in_channels , out_channels , kernel_size , stride , padding , bias = bias )
    weight_initialization( conv.weight ,  activation )
    layers.append( conv )
    if not preactivation and activation is not None:
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        layers.append( activation )
    return nn.Sequential( *layers )

def deconv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 ,  output_padding = 0 , activation = None ,   use_batchnorm = False , preactivation = False , bias= True):
    if use_batchnorm:
        bias = False

    layers = []
    if preactivation and activation is not None:
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        
        layers.append( activation )
    deconv = nn.ConvTranspose2d( in_channels , out_channels , kernel_size , stride ,  padding , output_padding , bias = bias )
    weight_initialization( deconv.weight , activation )
    layers.append( deconv )
    if not preactivation and activation is not None:
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        layers.append( activation )
    return nn.Sequential( *layers )

def linear( in_channels , out_channels , activation = None , use_batchnorm = False ,bias = True):
    layers = []
    layers.append( nn.Linear( in_channels , out_channels ) )
    if use_batchnorm:
        layers.append( nn.BatchNorm1d( out_channels ))
    if activation is not None:
        layers.append( activation )
    return nn.Sequential( *layers )

class BasicBlock(nn.Module):
    def __init__(self, in_channels , out_channels , stride = 1 , use_batchnorm = False , activation = nn.ReLU(inplace=True) , preactivation = False , scaling_factor = 1.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv( in_channels , out_channels , 3 , stride , 1 ,  activation , use_batchnorm )
        self.conv2 = conv( out_channels , out_channels , 3 , 1 , 1 , None , use_batchnorm )
        self.downsample = None
        if stride != 1 or in_channels != out_channels :
            self.downsample = conv( in_channels , out_channels , 1 , stride , 0 , None , use_batchnorm )
        self.activation = copy.deepcopy( activation )
        self.scaling_factor = scaling_factor
    def forward(self , x ):
        residual = x 
        if self.downsample is not None:
            residual = self.downsample( residual )

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual * self.scaling_factor
        out = self.activation( out )

        return out



