#wrappers for convenience
import torch.nn as nn
from torch.nn.init import xavier_normal , kaiming_normal

def sequential(*kargs ):
    seq = nn.Sequential(*kargs)
    for layer in reversed(kargs):
        if hasattr( layer , 'out_channels'):
            seq.out_channels = layer.out_channels
            break
        if hasattr( layer , 'out_features'):
            seq.out_channels = layer.out_features
            break
    return seq

def weight_initialization( weight , init , activation):
    if init is None:
        return
    if init == "kaiming":
        assert not activation is None
        if hasattr(activation,"negative_slope"):
            kaiming_normal( weight , a = activation.negative_slope )
        else:
            kaiming_normal( weight , a = 0 )
    elif init == "xavier":
        xavier_normal( weight )
    return

def conv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 , init = "kaiming" , activation = nn.ReLU() , use_batchnorm = False ):
    convs = []
    if type(padding) == type(list()) :
        assert len(padding) != 3 
        if len(padding)==4:
            convs.append( nn.ReflectionPad2d( padding ) )
            padding = 0

    #print(padding)
    convs.append( nn.Conv2d( in_channels , out_channels , kernel_size , stride , padding ) )
    #weight init
    weight_initialization( convs[-1].weight , init , activation )
    #activation
    if not activation is None:
        convs.append( activation )
    #bn
    if use_batchnorm:
        convs.append( nn.BatchNorm2d( out_channels ) )
    seq = nn.Sequential( *convs )
    seq.out_channels = out_channels
    return seq

def deconv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 ,  output_padding = 0 , init = "kaiming" , activation = nn.ReLU() , use_batchnorm = False):
    convs = []
    convs.append( nn.ConvTranspose2d( in_channels , out_channels , kernel_size , stride ,  padding , output_padding ) )
    #weight init
    weight_initialization( convs[0].weight , init , activation )
    #activation
    if not activation is None:
        convs.append( activation )
    #bn
    if use_batchnorm:
        convs.append( nn.BatchNorm2d( out_channels ) )
    seq = nn.Sequential( *convs )
    seq.out_channels = out_channels
    return seq

class ResidualBlock(nn.Module):
    def __init__(self, in_channels , 
                out_channels = None, 
                kernel_size = 3, 
                stride = 1,
                padding = None , 
                weight_init = "kaiming" , 
                activation = nn.ReLU() ,
                is_bottleneck = False ,
                use_projection = False,
                scaling_factor = 1.0
                ):
        super(type(self),self).__init__()
        if out_channels is None:
            out_channels = in_channels // stride
        self.out_channels = out_channels
        self.use_projection = use_projection
        self.scaling_factor = scaling_factor
        self.activation = activation

        convs = []
        assert stride in [1,2]
        if stride == 1 :
            self.shorcut = nn.Sequential()
        else:
            self.shorcut = conv( in_channels , out_channels , 1 , stride , 0 , None , None , False )
        if is_bottleneck:
            convs.append( conv( in_channels     , in_channels//2  , 1 , 1 , 0   , weight_init , activation , False))
            convs.append( conv( in_channels//2  , out_channels//2 , kernel_size , stride , (kernel_size - 1)//2 , weight_init , activation , False))
            convs.append( conv( out_channels//2 , out_channels    , 1 , 1 , 0 , None , None       , False))
        else:
            convs.append( conv( in_channels , in_channels  , kernel_size , 1 , padding if padding is not None else (kernel_size - 1)//2 , weight_init , activation , False))
            convs.append( conv( in_channels , out_channels , kernel_size , 1 , padding if padding is not None else (kernel_size - 1)//2 , None , None       , False))
        
        self.layers = nn.Sequential( *convs )
    def forward(self,x):
        return self.activation( self.layers(x) + self.scaling_factor * self.shorcut(x) )



