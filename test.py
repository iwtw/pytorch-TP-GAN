import torch
from data import TestDataset
import numpy as np
from network import Discriminator,Generator
from torch.autograd import Variable
import time 
from utils import *
import importlib
import argparse

test_time = False
    

def parse_args():
    parser = argparse.ArgumentParser( description = "bicubic" )
    parser.add_argument("-input_list")
    parser.add_argument('-landmark_list')
    parser.add_argument('-resume_model',help='resume_model dirname')
    parser.add_argument("-subdir",help='output_dir = save/$resume_model/test/$subdir')
    parser.add_argument('--batch_size',type=int , default = 256 )
    flag_parser = parser.add_mutually_exclusive_group(required=False)#whether the input images are in the format of label1/img1 label2/img2
    flag_parser.add_argument("--folder",dest='folder',action="store_true")
    flag_parser.add_argument("--nofolder",dest='folder',action='store_false')
    parser.set_defaults( folder= True )
    args = parser.parse_args()
    return args
def init_dir(args):
    os.system( 'mkdir -p {}'.format('/'.join([args.resume_model,'test',args.subdir,'single'])))
    os.system( 'mkdir -p {}'.format('/'.join([args.resume_model,'test',args.subdir,'grid'])))

if __name__ == "__main__":
    args = parse_args()
    init_dir(args)
    img_list = open(args.input_list,'r').read().split('\n')
    img_list.pop()
    lm_list = open(args.landmark_list,'r').read().split('\n')
    lm_list.pop()


    #input
    train_config = importlib.import_module(  '.'.join( [ *args.resume_model.split('/') , 'config']  ) )
    dataloader = torch.utils.data.DataLoader( TestDataset( img_list , lm_list ) , batch_size = args.batch_size , shuffle = False , num_workers = 8 , pin_memory = True) 

    G = Generator(zdim = train_config.G['zdim'], use_batchnorm = train_config.G['use_batchnorm'] , use_residual_block = train_config.G['use_residual_block'] , num_classes = train_config.G['num_classes']).cuda()
    D = Discriminator(use_batchnorm = train_config.D['use_batchnorm']).cuda()
    if args.resume_model is not None:
        e1 = resume_model( G , args.resume_model )
        e2 = resume_model( D , args.resume_model )
        assert e1 == e2
        

    set_requires_grad(G,False)
    set_requires_grad(D,False)



    for step,batch in enumerate(dataloader):
        if test_time:
            print("step : ", step)
            t_pre = time.time() 
            print("preprocess time : ",t_pre - tt )
            tt = t_pre
        for k in  batch:
            batch[k] =  Variable( batch[k].cuda(async = True) ) 
        left_eye_patch = batch['left_eye']
        right_eye_patch = batch['right_eye']
        nose_patch = batch['nose']
        mouth_patch = batch['mouth']
        img = batch['img']
        img32 = batch['img32']
        img64 = batch['img64']
        #img_frontal = batch['img_frontal']
        #label =  batch['label'] 
        #print(torch.min(img)[0] , torch.max(img)[0] )
        #print(torch.min(left_eye_patch)[0] , torch.max(left_eye_patch)[0] )
        z = Variable( torch.FloatTensor( np.random.uniform(-1,1,(len(batch['img']),train_config.G['zdim'])) ).cuda() )
        if test_time:
            t_mv_to_cuda = time.time()
            print("mv_to_cuda time : ",t_mv_to_cuda - tt )
            tt = t_mv_to_cuda

        img128_fake , img64_fake , img32_fake , G_encoder_outputs , local_predict , le_fake , re_fake , nose_fake , mouth_fake , local_input = G( batch['img'] , batch['img64'] , batch['img32'] ,  batch['left_eye'] , batch['right_eye'] , batch['nose'] , batch['mouth'] , z , use_dropout = False )
        if test_time:
            t_forward_G = time.time()
            print("forward_G time : ",t_forward_G - tt )
            tt = t_forward_G

        for i in range(img128_fake.shape[0]):
            img_name = img_list[step*args.batch_size+i].split('/')[-1]
            save_image(img128_fake[i].data , '/'.join([args.resume_model,'test',args.subdir,'single',img_name]) )
            #print(resize(right_eye_patch[i].data.cpu(),(128,128)).shape)
            save_image(torch.stack([img128_fake[i].data.cpu(),batch['img'][i].data.cpu(),local_predict.data.cpu() , local_input.data.cpu() ) , '/'.join([args.resume_model,'test',args.subdir,'grid',img_name]))

        if test_time:
            t_backward = time.time()
            print("backward time : ",t_backward - tt )
            tt = t_backward

        if test_time:
            t_numpy = time.time()
            print("numy time: " , t_numpy - tt )
            tt = t_numpy
                

            t = new_t

