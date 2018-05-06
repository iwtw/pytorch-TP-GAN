import torch
from torchvision import transforms
import config
from data import TrainDataset
import numpy as np
from skimage.io import imsave
from network import Discriminator,Generator
from torch.autograd import Variable
import time 
from log import TensorBoardX
from utils import *
import feature_extract_network
import importlib

test_time = False
    
if __name__ == "__main__":
    img_list = open(config.train['img_list'],'r').read().split('\n')
    img_list.pop()

    #input
    dataloader = torch.utils.data.DataLoader( TrainDataset( img_list ) , batch_size = config.train['batch_size'] , shuffle = True , num_workers = 8 , pin_memory = True) 

    G = torch.nn.DataParallel( Generator(zdim = config.G['zdim'], use_batchnorm = config.G['use_batchnorm'] , use_residual_block = config.G['use_residual_block'] , num_classes = config.G['num_classes'])).cuda()
    D = torch.nn.DataParallel( Discriminator(use_batchnorm = config.D['use_batchnorm'])).cuda()
    optimizer_G = torch.optim.Adam( filter(lambda p: p.requires_grad , G.parameters()) , lr = 1e-4)
    optimizer_D = torch.optim.Adam( filter(lambda p: p.requires_grad , D.parameters()) , lr = 1e-4)
    last_epoch = -1
    if config.train['resume_model'] is not None:
        e1 = resume_model( G , config.train['resume_model'] )
        e2 = resume_model( D , config.train['resume_model'] )
        assert e1 == e2
        last_epoch = e1
        
    if config.train['resume_optimizer'] is not None:
        e3 = resume_optimizer( optimizer_G , G ,  config.train['resume_optimizer'] )
        e4 = resume_optimizer( optimizer_D , D , config.train['resume_optimizer'] )
        assert e1==e2 and e2 == e3 and e3 == e4
        last_epoch = e1



    tb = TensorBoardX(config_filename_list = ["config.py" ] )


    #d = torch.load('./feature_extract_models/resnet18_finetune_MultiPie_epoch19.pth')
    #feature_extract_model = resnet.resnet18()
    #feature_extract_model.fc1 = torch.nn.Linear( 512*3*3 , 512 )
    pretrain_config = importlib.import_module( '.'.join( [ *config.feature_extract_model['resume'].split('/') ,  'pretrain_config' ]   ) )
    model_name = pretrain_config.stem['model_name']
    kwargs = pretrain_config.stem
    kwargs.pop('model_name')
    feature_extract_model = eval( 'feature_extract_network.' + model_name)(**kwargs)

    resume_model( feature_extract_model , config.feature_extract_model['resume'] , strict = True ) 
    feature_extract_model = torch.nn.DataParallel(  feature_extract_model ).cuda()

    l1_loss = torch.nn.L1Loss().cuda()
    mse = torch.nn.MSELoss().cuda()
    cross_entropy = torch.nn.CrossEntropyLoss().cuda()

    for param in feature_extract_model.parameters():
        param.requires_grad = False
    t = time.time()
    if test_time:
        tt = time.time()

    for epoch in range( last_epoch + 1  , config.train['num_epochs']):
        for step,batch in enumerate(dataloader):
            #if epoch==0:
            #    optimizer_G.param_groups[0]['lr'] = config.train['learning_rate'] * lr_warmup(  step , len(dataloader) )
            #    optimizer_D.param_groups[0]['lr'] = config.train['learning_rate'] * lr_warmup(  step , len(dataloader) )
            if test_time:
                print("step : ", step)
                t_pre = time.time() 
                print("preprocess time : ",t_pre - tt )
                tt = t_pre
            for k in  batch:
                batch[k] =  Variable( batch[k].cuda(async = True) , requires_grad = False ) 
            z = Variable( torch.FloatTensor( np.random.uniform(-1,1,(len(batch['img']),config.G['zdim'])) ).cuda() )
            if test_time:
                t_mv_to_cuda = time.time()
                print("mv_to_cuda time : ",t_mv_to_cuda - tt )
                tt = t_mv_to_cuda

            img128_fake , img64_fake , img32_fake , G_encoder_outputs , local_predict , le_fake , re_fake , nose_fake , mouth_fake , local_input = G( batch['img'] , batch['img64'] , batch['img32'] ,  batch['left_eye'] , batch['right_eye'] , batch['nose'] , batch['mouth'] , z , use_dropout = True )
            if test_time:
                t_forward_G = time.time()
                print("forward_G time : ",t_forward_G - tt )
                tt = t_forward_G

            set_requires_grad( D , True )

            # compute loss and backward
            #L_D = torch.mean( - torch.log( D(img_frontal)) - torch.log( 1 -  D(img128_fake.detach()))  )
            adv_D_loss = - torch.mean( D( batch['img_frontal'] )  ) + torch.mean( D( img128_fake.detach() ) )  
            #compute the gradient penalty
            alpha = torch.rand( batch['img_frontal'].shape[0] , 1 , 1 , 1 ).expand_as(batch['img_frontal']).pin_memory().cuda(async = True)
            interpolated_x = Variable( alpha * img128_fake.detach().data   + (1.0 - alpha) * batch['img_frontal'].data , requires_grad = True) 
            out = D(interpolated_x)
            dxdD = torch.autograd.grad( outputs = out , inputs = interpolated_x , grad_outputs = torch.ones(out.size()).cuda() , retain_graph = True , create_graph = True , only_inputs = True  )[0].view(out.shape[0],-1)
            gp_loss = torch.mean( ( torch.norm( dxdD , p = 2 ) - 1 )**2 )
            L_D = adv_D_loss + config.loss['weight_gradient_penalty'] * gp_loss


            optimizer_D.zero_grad()
            L_D.backward()
            optimizer_D.step()

            set_requires_grad( D , False )
            adv_G_loss = - torch.mean( D(img128_fake) )   
            pixelwise_128_loss = l1_loss( img128_fake , batch['img_frontal'])
            pixelwise_64_loss = l1_loss( img64_fake , batch['img64_frontal'])
            pixelwise_32_loss = l1_loss( img32_fake , batch['img32_frontal'])
            pixelwise_loss = config.loss['weight_128'] * pixelwise_128_loss + config.loss['weight_64'] * pixelwise_64_loss + config.loss['weight_32'] * pixelwise_32_loss

            eyel_loss = l1_loss( le_fake , batch['left_eye_frontal'] )
            eyer_loss = l1_loss( re_fake , batch['right_eye_frontal'] )
            nose_loss = l1_loss( nose_fake , batch['nose_frontal'] )
            mouth_loss = l1_loss( mouth_fake , batch['mouth_frontal'] )
            pixelwise_local_loss = eyel_loss + eyer_loss + nose_loss + mouth_loss



            inv_idx128 = torch.arange(img128_fake.size()[3]-1, -1, -1).long().cuda()
            img128_fake_flip = img128_fake.index_select(3, Variable( inv_idx128))
            img128_fake_flip.detach_()
            inv_idx64 = torch.arange(img64_fake.size()[3]-1, -1, -1).long().cuda()
            img64_fake_flip = img64_fake.index_select(3, Variable( inv_idx64))
            img64_fake_flip.detach_()
            inv_idx32 = torch.arange(img32_fake.size()[3]-1, -1, -1).long().cuda()
            img32_fake_flip = img32_fake.index_select(3, Variable( inv_idx32))
            img32_fake_flip.detach_()
            symmetry_128_loss  = l1_loss( img128_fake , img128_fake_flip ) 
            symmetry_64_loss  = l1_loss( img64_fake , img64_fake_flip ) 
            symmetry_32_loss  = l1_loss( img32_fake , img32_fake_flip ) 
            symmetry_loss = config.loss['weight_128'] * symmetry_128_loss + config.loss['weight_64'] * symmetry_64_loss + config.loss['weight_32'] * symmetry_32_loss



            feature_frontal , fc_frontal = feature_extract_model( batch['img_frontal'] )
            feature_predict , fc_predict = feature_extract_model( img128_fake )

            #ip_loss =  mse(  avgpool_predict , avgpool_frontal.detach() ) +  mse( fc1_predict  , fc1_frontal.detach()  ) 
            ip_loss = mse( feature_predict , feature_frontal.detach() )

            tv_loss = torch.mean( torch.abs(  img128_fake[:,:,:-1,:] - img128_fake[:,:,1:,:] ) )  + torch.mean(  torch.abs( img128_fake[:,:,:,:-1] - img128_fake[:,:,:,1:] ) )  

            cross_entropy_loss =  cross_entropy( G_encoder_outputs , batch['label'] ) 
            L_syn = config.loss['weight_pixelwise']*pixelwise_loss + config.loss['weight_pixelwise_local'] * pixelwise_local_loss + config.loss['weight_symmetry']*symmetry_loss + config.loss['weight_adv_G']*adv_G_loss + config.loss['weight_identity_preserving']*ip_loss + config.loss['weight_total_varation']*tv_loss
            L_G = L_syn + config.loss['weight_cross_entropy']*cross_entropy_loss 
            optimizer_G.zero_grad()
            L_G.backward()
            optimizer_G.step()

            if test_time:
                t_backward = time.time()
                print("backward time : ",t_backward - tt )
                tt = t_backward

            tb.add_scalar( "D_loss" , L_D.data.cpu().numpy() , epoch*len(dataloader) + step , 'train' )
            tb.add_scalar( "G_loss" , L_G.data.cpu().numpy() , epoch*len(dataloader) + step , 'train' )
            tb.add_scalar( "adv_D_loss" , adv_D_loss.data.cpu().numpy() , epoch*len(dataloader) + step , 'train' )
            tb.add_scalar( "pixelwise_loss" , pixelwise_loss.data.cpu().numpy() , epoch*len(dataloader) + step , 'train' )
            tb.add_scalar( "pixelwise_local_loss" , pixelwise_local_loss.data.cpu().numpy() , epoch*len(dataloader) + step , 'train' )
            tb.add_scalar( "symmetry_loss" , symmetry_loss.data.cpu().numpy() , epoch*len(dataloader) + step , 'train' )
            tb.add_scalar( "adv_G_loss" , adv_G_loss.data.cpu().numpy() , epoch*len(dataloader) + step , 'train' )
            tb.add_scalar( "identity_preserving_loss" , ip_loss.data.cpu().numpy() , epoch*len(dataloader) + step , 'train')
            tb.add_scalar( "total_variation_loss" , tv_loss.data.cpu().numpy() , epoch*len(dataloader) + step , 'train')
            tb.add_scalar( "cross_entropy_loss" , cross_entropy_loss.data.cpu().numpy() , epoch*len(dataloader) + step , 'train')
            if test_time:
                t_numpy = time.time()
                print("numy time: " , t_numpy - tt )
                tt = t_numpy
                    

            if step% config.train['log_step'] == 0 :
                new_t = time.time()
                print( "epoch {} , step {} / {} , adv_D_loss {:.3f} , gradient_penalty_loss {:.3f} , G_loss {:.3f} , pixelwise_loss {:.3f} , pixelwise_local_loss {:.3f} , symmetry_loss {:.3f} , adv_G_loss {:.3f} , identity_preserving_loss {:.3f} , total_variation_loss {:.3f} , cross_entropy_loss {:.3f} ,  {:.1f} imgs/s".format( epoch , step , len(dataloader ) , adv_D_loss.data.cpu().numpy()[0] , gp_loss.data.cpu().numpy()[0] , L_G.data.cpu().numpy()[0] ,  pixelwise_loss.data.cpu().numpy()[0] , pixelwise_local_loss.data.cpu().numpy()[0] , symmetry_loss.data.cpu().numpy()[0] , adv_G_loss.data.cpu().numpy()[0] , ip_loss.data.cpu().numpy()[0] , tv_loss.data.cpu().numpy()[0] , cross_entropy_loss.data.cpu().numpy()[0] , config.train['log_step']*config.train['batch_size'] / ( new_t - t ) ) )
                tb.add_image_grid( "grid/predict" , 4 , img128_fake.data.float() / 2.0 + 0.5 , epoch*len(dataloader) + step , 'train')
                tb.add_image_grid( "grid/frontal" , 4 , batch['img_frontal'].data.float() / 2.0 + 0.5 , epoch*len(dataloader) + step , 'train')
                tb.add_image_grid( "grid/profile" , 4 , batch['img'].data.float() / 2.0 + 0.5 , epoch*len(dataloader) + step , 'train')
                tb.add_image_grid( "grid/local" , 4 , local_predict.data.float() / 2.0 + 0.5 , epoch*len(dataloader) + step  , 'train' )
                tb.add_image_grid( "grid/local_input" , 4 , local_input.data.float() / 2.0 + 0.5 , epoch*len(dataloader) + step  , 'train' )
                #tb.add_image_grid( "grid/left_eye" , 4 , left_eye_patch.data.float() / 2.0 + 0.5 , epoch* len( dataloader) + step , 'train')
                t = new_t
        #epoch end

        save_model(G,tb.path,epoch)
        save_model(D,tb.path,epoch)
        save_optimizer(optimizer_G,G, tb.path,epoch)
        save_optimizer(optimizer_D,D, tb.path,epoch)
        print( "Save done at {}".format(tb.path) )

