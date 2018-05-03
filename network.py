import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from layers import *
from utils import resize,elementwise_mult_cast_int

emci = elementwise_mult_cast_int
class LocalPathway(nn.Module):
    def __init__(self,use_batchnorm = True,feature_layer_dim = 64 , fm_mult = 1.0):
        super(LocalPathway,self).__init__()
        n_fm_encoder = [64,128,256,512] 
        n_fm_decoder = [256,128] 
        n_fm_encoder = emci(n_fm_encoder,fm_mult)
        n_fm_decoder = emci(n_fm_decoder,fm_mult)
        #encoder
        self.conv0 = sequential( conv( 3   , n_fm_encoder[0]  , 3 , 1 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm) ,
                                ResidualBlock(n_fm_encoder[0] , activation = nn.LeakyReLU() ) )
        self.conv1 = sequential( conv( n_fm_encoder[0]  , n_fm_encoder[1] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm) ,
                                ResidualBlock(n_fm_encoder[1] , activation = nn.LeakyReLU() ) )
        self.conv2 = sequential( conv( n_fm_encoder[1] , n_fm_encoder[2] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm) ,
                                ResidualBlock(n_fm_encoder[2] , activation = nn.LeakyReLU() ) )
        self.conv3 = sequential( conv( n_fm_encoder[2] , n_fm_encoder[3] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm) ,
                                ResidualBlock(n_fm_encoder[3] , activation = nn.LeakyReLU() ) )
        #decoder
        self.deconv0 =   deconv( n_fm_encoder[3] , n_fm_decoder[0] , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm) 
        self.after_select0 =  sequential(   conv( n_fm_decoder[0] + self.conv2.out_channels , n_fm_decoder[0] , 3 , 1 , 1 , 'kaiming' ,  nn.LeakyReLU() , use_batchnorm  ) ,    ResidualBlock( n_fm_decoder[0] , activation = nn.LeakyReLU()  )  )

        self.deconv1 =   deconv( self.after_select0.out_channels , n_fm_decoder[1] , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm) 
        self.after_select1 = sequential(    conv( n_fm_decoder[1] + self.conv1.out_channels , n_fm_decoder[1] , 3 , 1 , 1 , 'kaiming' , nn.LeakyReLU() , use_batchnorm  ) ,   ResidualBlock( n_fm_decoder[1] , activation = nn.LeakyReLU()  )  )

        self.deconv2 =   deconv( self.after_select1.out_channels , feature_layer_dim , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm) 
        self.after_select2 = sequential( conv( feature_layer_dim + self.conv0.out_channels , feature_layer_dim  , 3 , 1 , 1 , 'kaiming' , nn.LeakyReLU() , use_batchnorm  ) ,   ResidualBlock( feature_layer_dim , activation = nn.LeakyReLU()  )  )
        self.local_img = conv( feature_layer_dim  , 3 , 1 , 1 , 0 , None , None , False )

                            
    def forward(self,x):
        conv0 = self.conv0( x )
        conv1 = self.conv1( conv0 )
        conv2 = self.conv2( conv1 )
        conv3 = self.conv3( conv2 )
        deconv0 = self.deconv0( conv3 )
        after_select0 = self.after_select0( torch.cat([deconv0,conv2],  1) )
        deconv1 = self.deconv1( after_select0 )
        after_select1 = self.after_select1( torch.cat([deconv1,conv1] , 1) )
        deconv2 = self.deconv2( after_select1 )
        after_select2 = self.after_select2( torch.cat([deconv2,conv0],  1 ) )
        local_img = self.local_img( after_select2 )
        assert local_img.shape == x.shape  ,  "{} {}".format(local_img.shape , x.shape)
        return  local_img , deconv2

class LocalFuser(nn.Module):
    #differs from original code here
    #https://github.com/HRLTY/TP-GAN/blob/master/TP_GAN-Mar6FS.py
    '''
    x         y
    39.4799 40.2799
    85.9613 38.7062
    63.6415 63.6473
    45.6705 89.9648
    83.9000 88.6898
    this is the mean locaiton of 5 landmarks
    '''
    def __init__(self ):
        super(LocalFuser,self).__init__()
    def forward( self , f_left_eye , f_right_eye , f_nose , f_mouth):
        EYE_W , EYE_H = 40 , 40 
        NOSE_W , NOSE_H = 40 , 32
        MOUTH_W , MOUTH_H = 48 , 32
        IMG_SIZE = 128
        f_left_eye = torch.nn.functional.pad(f_left_eye , (39 - EYE_W//2  - 1 ,IMG_SIZE - (39 + EYE_W//2 - 1) ,40 - EYE_H//2 - 1, IMG_SIZE - (40 + EYE_H//2 - 1)))
        f_right_eye = torch.nn.functional.pad(f_right_eye,(86 - EYE_W//2  - 1 ,IMG_SIZE - (86 + EYE_W//2 - 1) ,39 - EYE_H//2 - 1, IMG_SIZE - (39 + EYE_H//2 - 1)))
        f_nose = torch.nn.functional.pad(f_nose,          (64 - NOSE_W//2 - 1 ,IMG_SIZE - (64 + NOSE_W//2 -1) ,64 - NOSE_H//2- 1, IMG_SIZE - (64 + NOSE_H//2- 1)))
        f_mouth = torch.nn.functional.pad(f_mouth,        (84 - MOUTH_W//2 -1 ,IMG_SIZE - (84 + MOUTH_W//2 -1),89 - MOUTH_H//2-1, IMG_SIZE - (89 + MOUTH_H//2-1)))
        return torch.max( torch.stack( [ f_left_eye , f_right_eye , f_nose , f_mouth] , dim = 0  ) , dim = 0 )[0]

class GlobalPathway(nn.Module):
    def __init__(self, zdim , local_feature_layer_dim = 64 , use_batchnorm = True , use_residual_block = True , scaling_factor = 1.0 , fm_mult = 1.0):
        super(GlobalPathway,self).__init__()
        n_fm_encoder = [64,64,128,256,512]   
        n_fm_decoder_initial = [64,32,16,8] 
        n_fm_decoder_reconstruct = [512,256,128,64]
        n_fm_decoder_conv = [64,32]
        n_fm_encoder = emci(n_fm_encoder , fm_mult)
        n_fm_decoder_initial = emci( n_fm_decoder_initial , fm_mult )
        n_fm_decoder_reconstruct = emci( n_fm_decoder_reconstruct , fm_mult )
        n_fm_decoder_conv = emci( n_fm_decoder_conv , fm_mult )

        

        self.zdim = zdim
        self.use_residual_block = use_residual_block
        #encoder
        #128x128
        self.conv0 = sequential( conv( 3   , n_fm_encoder[0]  , 7 , 1 , 3 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    ResidualBlock( 64 , 64 , 7 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , scaling_factor = scaling_factor)
                                  )
        #64x64
        self.conv1 = sequential( conv( n_fm_encoder[1]  , n_fm_encoder[1]  , 5 , 2 , 2 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    ResidualBlock( 64 , 64 , 5 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , scaling_factor = scaling_factor)
                                  )
        #32x32
        self.conv2 = sequential( conv( n_fm_encoder[1]  , n_fm_encoder[2] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    ResidualBlock( 128 , 128 , 3 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , scaling_factor = scaling_factor)
                                  )
        #16x16
        self.conv3 = sequential( conv( n_fm_encoder[2] , n_fm_encoder[3] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    ResidualBlock( 256 , 256 , 3 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , is_bottleneck = False , scaling_factor = scaling_factor)
                                  )
        #8x8
        self.conv4 = sequential( conv( n_fm_encoder[3] , n_fm_encoder[4] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    *[ ResidualBlock( 512 , 512 , 3 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , is_bottleneck = False , scaling_factor = scaling_factor) for i in range(4) ]
                                  )
        self.fc1 = nn.Linear( n_fm_encoder[4]*8*8 , 512)
        self.fc2 = nn.MaxPool1d( 2 , 2 , 0)
        torch.nn.functional.max_pool1d
        #decoder
        self.initial_8    = deconv( 256 + self.zdim , n_fm_decoder_initial[0] , 8 , 1 , 0 , 0 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.initial_32   = deconv( n_fm_decoder_initial[0] , n_fm_decoder_initial[1] , 3 , 4 , 0 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.initial_64   = deconv( n_fm_decoder_initial[1] , n_fm_decoder_initial[2] , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.initial_128  = deconv( n_fm_decoder_initial[2] , n_fm_decoder_initial[3] , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)

        dim8 = self.initial_8.out_channels + self.conv4.out_channels
        self.before_select_8 = ResidualBlock( dim8 , dim8 , 3 , activation = nn.LeakyReLU() )
        self.reconstruct_8 = sequential( *[ResidualBlock( dim8 , activation = nn.LeakyReLU() ) for i in range(2)] )

        
        self.reconstruct_deconv_16 = deconv( self.reconstruct_8.out_channels , n_fm_decoder_reconstruct[0] , 3 , 2 , 1 , 1, 'kaiming' , nn.ReLU() , use_batchnorm )
        dim16 = self.conv3.out_channels
        self.before_select_16 = ResidualBlock( dim16 , activation =nn.LeakyReLU() )
        self.reconstruct_16 = sequential( *[ResidualBlock( self.reconstruct_deconv_16.out_channels + self.before_select_16.out_channels , activation = nn.LeakyReLU() )for i in range(2)])

        self.reconstruct_deconv_32 = deconv( self.reconstruct_16.out_channels , n_fm_decoder_reconstruct[1] , 3 , 2 , 1 , 1, 'kaiming' , nn.ReLU() , use_batchnorm )
        dim32 = self.conv2.out_channels + self.initial_32.out_channels + 3
        self.before_select_32 = ResidualBlock( dim32 , activation = nn.LeakyReLU() )
        self.reconstruct_32 = sequential( *[ResidualBlock( dim32 + n_fm_decoder_reconstruct[1]   , activation = nn.LeakyReLU()) for i in range(2) ]  )
        self.decoded_img32 = conv( self.reconstruct_32.out_channels , 3 , 1 , 1 , 0 , None ,  None )

        self.reconstruct_deconv_64 = deconv( self.reconstruct_32.out_channels , n_fm_decoder_reconstruct[2] , 3 , 2 , 1 , 1 , 'kaiming' , nn.ReLU() , use_batchnorm )
        dim64 = self.conv1.out_channels + self.initial_64.out_channels + 3
        self.before_select_64 = ResidualBlock(  dim64 , kernel_size =  5 , activation = nn.LeakyReLU()   ) 
        self.reconstruct_64 = sequential( *[ResidualBlock( dim64 + n_fm_decoder_reconstruct[2] + 3 , activation = nn.LeakyReLU()) for i in range(2)])
        self.decoded_img64 = conv( self.reconstruct_64.out_channels , 3 , 1 , 1 , 0 , None ,  None )

        self.reconstruct_deconv_128 = deconv( self.reconstruct_64.out_channels , n_fm_decoder_reconstruct[3] , 3 , 2 , 1 , 1 , 'kaiming' , nn.ReLU() , use_batchnorm )
        dim128 = self.conv0.out_channels + self.initial_128.out_channels + 3
        self.before_select_128 = ResidualBlock( dim128  , kernel_size = 7 , activation = nn.LeakyReLU()  )
        self.reconstruct_128 = sequential( *[ResidualBlock( dim128 + n_fm_decoder_reconstruct[3] + local_feature_layer_dim + 3 + 3 , kernel_size = 5 , activation = nn.LeakyReLU())] )
        self.conv5 = sequential( conv( self.reconstruct_128.out_channels , n_fm_decoder_conv[0] , 5 , 1 , 2 , 'kaiming' , nn.LeakyReLU() , use_batchnorm  ) , \
                ResidualBlock(n_fm_decoder_conv[0] , kernel_size = 3 , activation = nn.LeakyReLU() ))
        self.conv6 = conv( n_fm_decoder_conv[0] , n_fm_decoder_conv[1] , 3 , 1 , 1 , 'kaiming' , nn.LeakyReLU() , use_batchnorm )
        self.decoded_img128 = conv( n_fm_decoder_conv[1] , 3 , 1 , 1 , 0 , None , activation = None )

    def forward(self, I128 , I64 , I32 ,  local_predict , local_feature , z ):
        #encoder
        conv0 = self.conv0( I128)#128x128
        conv1 = self.conv1( conv0)#64x64
        conv2 = self.conv2( conv1)#32x32
        conv3 = self.conv3( conv2)#16x16
        conv4 = self.conv4( conv3)#8x8

        fc1 = self.fc1( conv4.view( conv4.size()[0] , -1 ))
        fc2 = self.fc2( fc1.view( fc1.size()[0] , -1 , 2  )).view( fc1.size()[0] , -1 ) 

        #decoder
        initial_8   = self.initial_8( torch.cat([fc2,z] , 1).view( fc2.size()[0] , -1 , 1 , 1 )  )
        initial_32  = self.initial_32( initial_8)
        initial_64  = self.initial_64( initial_32)
        initial_128 = self.initial_128( initial_64)
        
        before_select_8 = self.before_select_8( torch.cat( [initial_8,conv4] , 1 ) )
        reconstruct_8 = self.reconstruct_8( before_select_8 )
        assert reconstruct_8.shape[2] == 8

        reconstruct_deconv_16 = self.reconstruct_deconv_16( reconstruct_8 )
        before_select_16 = self.before_select_16( conv3 )
        reconstruct_16 = self.reconstruct_16( torch.cat( [reconstruct_deconv_16 , before_select_16] , 1 ) )
        assert reconstruct_16.shape[2] == 16

        reconstruct_deconv_32 = self.reconstruct_deconv_32( reconstruct_16 )
        before_select_32 = self.before_select_32( torch.cat( [initial_32 , conv2 , I32] ,  1 ) )
        reconstruct_32 = self.reconstruct_32( torch.cat( [reconstruct_deconv_32,before_select_32] , 1 ) )
        decoded_img32 = self.decoded_img32( reconstruct_32 )
        assert decoded_img32.shape[2] == 32

        reconstruct_deconv_64 = self.reconstruct_deconv_64( reconstruct_32 )
        before_select_64 = self.before_select_64( torch.cat( [initial_64 , conv1 , I64] , 1 ) )
        reconstruct_64 = self.reconstruct_64( torch.cat( [reconstruct_deconv_64 , before_select_64 , torch.nn.functional.upsample( decoded_img32.data, (64,64) , mode = 'bilinear') ] , 1))
        decoded_img64 = self.decoded_img64( reconstruct_64 )
        assert decoded_img64.shape[2] == 64

        reconstruct_deconv_128 = self.reconstruct_deconv_128( reconstruct_64 )
        before_select_128 = self.before_select_128( torch.cat( [initial_128 , conv0 , I128 ] , 1 ) )
        reconstruct_128 = self.reconstruct_128( torch.cat( [reconstruct_deconv_128 , before_select_128 , torch.nn.functional.upsample(decoded_img64 , (128,128) , mode = 'bilinear' ) ,  local_feature , local_predict ] , 1 ) )
        conv5 = self.conv5( reconstruct_128 )
        conv6 = self.conv6( conv5 )
        decoded_img128 = self.decoded_img128( conv6 )
        return decoded_img128 , decoded_img64 , decoded_img32 , fc2
        
class FeaturePredict(nn.Module):
    def __init__(self ,  num_classes , global_feature_layer_dim = 256 , dropout = 0.3):
        super(FeaturePredict,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(global_feature_layer_dim , num_classes )
    def forward(self ,x ,use_dropout):
        if use_dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x
        
class Generator(nn.Module):
    def __init__(self, zdim , num_classes , use_batchnorm = True , use_residual_block = True):
        super(Generator,self).__init__()
        self.local_pathway_left_eye  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_pathway_right_eye  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_pathway_nose  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_pathway_mouth  = LocalPathway(use_batchnorm = use_batchnorm)

        self.global_pathway = GlobalPathway(zdim , use_batchnorm = use_batchnorm , use_residual_block = use_residual_block)
        self.local_fuser    = LocalFuser()
        self.feature_predict = FeaturePredict(num_classes)

    def forward( self, I128 , I64 , I32 , left_eye , right_eye , nose , mouth , z , use_dropout ):

        #pass through local pathway
        le_fake , le_fake_feature = self.local_pathway_left_eye( left_eye)
        re_fake , re_fake_feature = self.local_pathway_right_eye( right_eye)
        nose_fake , nose_fake_feature = self.local_pathway_nose( nose)
        mouth_fake , mouth_fake_feature = self.local_pathway_mouth( mouth)

        #fusion
        local_feature = self.local_fuser( le_fake_feature , re_fake_feature , nose_fake_feature , mouth_fake_feature )
        local_vision = self.local_fuser( le_fake , re_fake , nose_fake , mouth_fake )
        local_input = self.local_fuser( left_eye , right_eye , nose , mouth )


        #pass through global pathway
        #fc1 , I_fake = self.global_pathway( I128 , I64 , I32 , local ,z)
        I128_fake , I64_fake , I32_fake , encoder_feature = self.global_pathway( I128 , I64 , I32 , local_vision , local_feature , z)
        encoder_predict = self.feature_predict( encoder_feature , use_dropout )
        
        return I128_fake , I64_fake , I32_fake , encoder_predict , local_vision , le_fake , re_fake , nose_fake , mouth_fake , local_input


        


class Discriminator(nn.Module):

    def __init__(self, use_batchnorm = False , fm_mult = 1.0):
        super(Discriminator,self).__init__()
        layers = []
        n_fmap = [3,64,128,256,512,512] 
        n_fmap = emci( n_fmap , fm_mult )
        for i in range( 5 ):
            #layers.append( conv( n_fmap[i] , n_fmap[i+1] , kernel_size = 4 , stride = 2 , padding = 1 , init = "kaiming" , activation = nn.LeakyReLU(1e-2) ) )
            layers.append( conv( n_fmap[i] , n_fmap[i+1] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm  ) )
            if i >=3:
                layers.append( ResidualBlock( n_fmap[i+1] , activation = nn.LeakyReLU() ) )
    
        layers.append( conv( n_fmap[-1] , 1 , kernel_size = 3 ,  stride = 1 , padding = 1 , init = None ,activation =  None ))
        self.model = sequential( *layers )

    def forward(self,x):
        return self.model(x)

