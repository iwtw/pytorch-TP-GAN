import torch.nn as nn
from feature_extract_layers import *
from MobileNetV2 import MobileNetV2
import copy

class ResNet( nn.Module ):
    def __init__(self,block,num_blocks,num_features,num_classes,use_batchnorm=True,feature_layer_dim=None,dream=None,activation=nn.ReLU(inplace=True),preactivation=False ,use_maxpool = True , use_avgpool = True , dropout = 0.0 ):
        'resolution : height 128 , width 128'
        super(ResNet,self).__init__()
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        self.preactivation = preactivation
        self.use_maxpool = use_maxpool
        self.use_avgpool = use_avgpool
        assert len(num_features) == 4 
        assert len(num_blocks) == 4 
        self.conv1 = conv( 3 , num_features[0] , 7 , 2 , 3, activation , use_batchnorm = use_batchnorm , bias = False  )
        if self.use_maxpool:
            self.maxpool = nn.MaxPool2d( 3,2,1)

        blocks = []
        blocks.append( self.build_blocks(block,num_features[0],num_features[0], 1 ,num_blocks[0]) )
        for i in range( 1,len(num_blocks)):
            blocks.append( self.build_blocks(block,num_features[i-1],num_features[i] , 2 , num_blocks[i] ) )
        self.blocks = nn.Sequential( *blocks )
        if self.preactivation:
            self.post_bn = nn.BatchNorm2d( num_features[-1] )

        if self.use_maxpool:
            shape = ( 4 , 4 )
        else:
            shape = ( 8 , 8 )
        if use_avgpool:
            self.avgpool = nn.AvgPool2d([*shape],1)
            shape = 1*1
        else:
            shape =  shape[0] * shape[1]

        if feature_layer_dim is not None:
            self.fc1 = linear( num_features[-1] * shape , feature_layer_dim ,use_batchnorm = use_batchnorm)
        self.dropout = nn.Dropout( dropout )

        self.fc2 = linear( feature_layer_dim if feature_layer_dim is not None else num_features[-1] * shape , num_classes , use_batchnorm = False )

    def build_blocks(self,block,in_channels,out_channels,stride,length):
        layers = []
        layers.append( block( in_channels , out_channels , stride , self.use_batchnorm , copy.deepcopy( self.activation),preactivation=self.preactivation ) )
        for i in range(1,length):
            layers.append( block(out_channels,out_channels, 1 , self.use_batchnorm , copy.deepcopy(self.activation) , preactivation = self.preactivation ) )
        return nn.Sequential( *layers )

    def forward(self,x,use_dropout=False):

        out = self.conv1(x)
        if self.use_maxpool:
            out = self.maxpool(out)
        out = self.blocks(out)
        if self.preactivation:
            out = self.post_bn( out )

        if self.use_avgpool:
            out = self.avgpool(out)
        
        if hasattr( self , 'fc1' ) :
            fc1 = self.fc1( out.view(out.shape[0],-1))
        else:
            fc1 = out

        fc1 = fc1.view( fc1.shape[0] , -1)
        fc2 = self.fc2( self.dropout(fc1) if use_dropout else fc1 )
        return fc2 , fc1

def resnet18( fm_mult = 1.0 , **kwargs ):
    num_features = [64,128,256,512]
    for i in range(len(num_features)):
        num_features[i] = int( num_features[i] * fm_mult )
    model = ResNet(BasicBlock, [2,2,2,2] , num_features , **kwargs ) 
    return model

def mobilenetv2(  **kwargs ):
    return MobileNetV2(  **kwargs )



    


