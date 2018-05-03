import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from layers import *

class LocalPathway(nn.Module):
    def __init__(self,use_batchnorm = True):
        super(LocalPathway,self).__init__()
        #encoder
        self.conv0 = conv( 3   , 64  , 3 , 1 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm)
        self.conv1 = conv( 64  , 128 , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm)
        self.conv2 = conv( 128 , 256 , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm)
        self.conv3 = conv( 256 , 512 , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm)
        #decoder
        self.deconv0 = deconv( 512 , 256 , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.deconv1 = deconv( 512 , 128 , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.deconv2 = deconv( 256 , 64  , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.conv4 = conv( 128  , 64  , 3 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.conv5 = conv( 64 ,  3   , 3 , 1 , 1 , "kaiming" , nn.ReLU() , False)
    def forward(self,x):
        conv0 = self.conv0( x )
        conv1 = self.conv1( conv0 )
        conv2 = self.conv2( conv1 )
        conv3 = self.conv3( conv2 )
        deconv0 = self.deconv0( conv3 )
        deconv1 = self.deconv1( torch.cat( [deconv0,conv2] , 1 ))
        deconv2 = self.deconv2( torch.cat( [deconv1,conv1] , 1 ))
        conv4 = self.conv4( torch.cat( [deconv2,conv0] , 1 ))
        conv5 = self.conv5( conv4 )
        return conv4 , conv5

class LocalFuser(nn.Module):
    def __init__(self ):
        super(LocalFuser,self).__init__()
    def forward( self , f_left_eye , f_right_eye , f_nose , f_mouth):
        f_left_eye = torch.nn.functional.pad(f_left_eye ,(24,64,24,64))
        f_right_eye = torch.nn.functional.pad(f_right_eye,(64,24,24,64))
        f_nose = torch.nn.functional.pad(f_nose,(44,44,48,48))
        f_mouth = torch.nn.functional.pad(f_mouth,(40,40,70,26))
        return torch.max( torch.stack( [ f_left_eye , f_right_eye , f_nose , f_mouth] , dim = 0  ) , dim = 0 )[0]

class GlobalPathway(nn.Module):
    def __init__(self, zdim , use_batchnorm = True , use_residual_block = True , scaling_factor = 1.0):
        super(GlobalPathway,self).__init__()
        self.zdim = zdim
        self.use_residual_block = use_residual_block
        #encoder
        #128x128
        self.conv0 = nn.Sequential( conv( 3   , 64  , 7 , 1 , 3 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    #ResidualBlock( 64 , 64 , 7 , None , "kaiming" , nn.LeakyReLU(1e-2) , scaling_factor = scaling_factor)
                                  )
        #64x64
        self.conv1 = nn.Sequential( conv( 64  , 64  , 5 , 2 , 2 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    #ResidualBlock( 64 , 64 , 5 , None , "kaiming" , nn.LeakyReLU(1e-2) , scaling_factor = scaling_factor)
                                  )
        #32x32
        self.conv2 = nn.Sequential( conv( 64  , 128 , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    #ResidualBlock( 128 , 128 , 3 , None , "kaiming" , nn.LeakyReLU(1e-2) , scaling_factor = scaling_factor)
                                  )
        #16x16
        self.conv3 = nn.Sequential( conv( 128 , 256 , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    #ResidualBlock( 256 , 256 , 3 , None , "kaiming" , nn.LeakyReLU(1e-2) , is_bottleneck = False , scaling_factor = scaling_factor)
                                  )
        #8x8
        self.conv4 = nn.Sequential( conv( 256 , 512 , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    #*[ ResidualBlock( 512 , 512 , 3 , None , "kaiming" , nn.LeakyReLU(1e-2) , is_bottleneck = False , scaling_factor = scaling_factor) for i in range(4) ]
                                  )
        self.fc1 = nn.Linear( 512*8*8 , 512)
        self.fc2 = nn.MaxPool1d( 2 , 2 , 0)
        torch.nn.functional.max_pool1d
        #decoder
        self.deconv0  = deconv( 256 + self.zdim , 64  , 8 , 1 , 0 , 0 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.deconv1  = deconv( 64  , 32  , 3 , 4 , 0 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.deconv2  = deconv( 32  , 16  , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.deconv3  = deconv( 16  , 8   , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)



    def forward(self, I , I64 , I32 ,  local , z ):
        #encoder
        conv0 = self.conv0( I)
        conv1 = self.conv1( conv0)
        conv2 = self.conv2( conv1)
        conv3 = self.conv3( conv2)
        conv4 = self.conv4( conv3)

        fc1 = self.fc1( conv4.view( conv4.size()[0] , -1 ))
        fc2 = self.fc2( fc1.view( fc1.size()[0] , -1 , 2  )).view( fc1.size()[0] , -1 ) 

        #decoder
        feat8   = self.feat8( torch.cat([fc2,z] , 1).view( fc2.size()[0] , -1 , 1 , 1 )  )
        feat32  = self.feat32( feat8)
        feat64  = self.feat64( feat32)
        feat128 = self.feat128( feat64)
        
        f_feat8   = feat8
        f_feat32  = feat32
        f_feat64  = feat64
        f_feat128 = feat128
        f_conv0 = conv0
        f_conv1 = conv1
        f_conv2 = conv2
        f_conv3 = conv3

        f32_I   = I32 
        f64_I   = I64
        f128_I  = I
        f_local = local

        #decoder deconv stack forward
        deconv0 = self.deconv0( torch.cat( [ f_feat8  , conv4] , 1))
        deconv1 = self.deconv1( torch.cat( [ f_conv3  , deconv0] , 1))
        deconv2 = self.deconv2( torch.cat( [ f_feat32 , f_conv2 , f32_I , deconv1 ] , 1))
        deconv3 = self.deconv3( torch.cat( [ f_feat64 , f_conv1 , f64_I , deconv2 ] , 1))
        conv5 = self.conv5( torch.cat( [ f_feat128 , f_conv0 , f_local ,f128_I , deconv3 ] , 1))
        conv6 = self.conv6( conv5)
        output = self.conv7( conv6)
        return fc1 , output
        
class Generator(nn.Module):
    def __init__(self, zdim , use_batchnorm = True , use_residual_block = True):
        super(Generator,self).__init__()
        self.local_pathway_left_eye  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_pathway_right_eye  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_pathway_nose  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_pathway_mouth  = LocalPathway(use_batchnorm = use_batchnorm)

        self.global_pathway = GlobalPathway(zdim , use_batchnorm = use_batchnorm , use_residual_block = use_residual_block)
        self.local_fuser    = LocalFuser()

    def forward( self, I , I64 , I32 , left_eye , right_eye , nose , mouth , z):

        #pass through local pathway
        f_le_fake , le_fake = self.local_pathway_left_eye( left_eye)
        f_re_fake , re_fake = self.local_pathway_right_eye( right_eye)
        f_nose_fake , nose_fake = self.local_pathway_nose( nose)
        f_mouth_fake , mouth_fake = self.local_pathway_mouth( mouth)

        #fusion
        local = self.local_fuser( f_le_fake , f_re_fake , f_nose_fake , f_mouth_fake)
        local_vision = self.local_fuser( le_fake , re_fake , nose_fake , mouth_fake)


        #pass through global pathway
        #fc1 , I_fake = self.global_pathway( I , I64 , I32 , local ,z)
        fc1 , I_fake = self.global_pathway( I , I64 , I32 , local_vision ,z)
        return fc1 , I_fake , local_vision


        


class Discriminator(nn.Module):
    # author of TPGAN did not mention the detailed network of Discriminator
    # but from the network graph in the paper , we can infer that there is
    # 6 conv in D and they all have same stirde of 2
    # it looks an awful lot like the one in "StarGAN" (which is adapted from PatchGAN)
    # so I adopt StarGAN's D 

    def __init__(self, use_batchnorm = False):
        super(Discriminator,self).__init__()
        self.use_batchnorm = use_batchnorm
        layers = []
        n_fmap = [3,64,128,256,512,1024]
        for i in range( 5 ):
            layers.append( conv( n_fmap[i] , n_fmap[i+1] , kernel_size = 4 , stride = 2 , padding = 1 , init = "kaiming" , activation = nn.LeakyReLU(1e-2) ) )
        layers.append( conv( n_fmap[-1] , 1 , kernel_size = 3 ,  stride = 1 , padding = 1 , init = None ,activation =  None ))
        if self.use_batchnorm:
            layers.append( nn.BatchNorm2d(1))
        self.model = nn.Sequential( *layers )

    def forward(self,x):
        return self.model(x)

