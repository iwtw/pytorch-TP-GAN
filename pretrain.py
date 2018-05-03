import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import feature_extract_network 
import numpy as np
from log import TensorBoardX
from math import pi
from utils import *
import pretrain_config as config
from data import PretrainDataset
import copy


def compute_loss(  predicts , labels  ):
    assert predicts.shape[0] == labels.shape[0]
    acc = torch.sum( torch.eq(labels  , torch.max( predicts , 1 )[1] ).long() ).float()/ float( labels.shape[0] )

    loss = cross_entropy( predicts , labels )
    return acc ,  loss
if __name__ == "__main__":
    tb = TensorBoardX(config_filename_list = ['pretrain_config.py'] , sub_dir = config.train['sub_dir'] +'/' + config.stem['model_name'] )
    log_file = open('/'.join( [tb.path,'train','log.txt'] ) , 'w' )

    train_img_list = open(config.train['train_img_list'],'r').read().split('\n')
    train_img_list.pop()
    val_img_list = open(config.train['val_img_list'],'r').read().split('\n')
    val_img_list.pop()

    train_dataset = PretrainDataset( train_img_list ) 
    val_dataset = PretrainDataset( val_img_list )
    train_dataloader = torch.utils.data.DataLoader(  train_dataset , batch_size = config.train['batch_size'] , shuffle = True , drop_last = True , num_workers = 8 , pin_memory = True) 
    val_dataloader = torch.utils.data.DataLoader(  val_dataset , batch_size = 30 , shuffle = True , drop_last = True , num_workers = 4 , pin_memory = True) 


    
    #if config.stem['model_name'] == 'resnet18':
    #    stem = feature_extract_network.resnet18( fm_mult = config.stem['fm_mult'] , num_classes = config.stem['num_classes'] , feature_layer_dim = config.stem['feature_layer_dim'] , use_batchnorm  = config.stem['use_batchnorm'] ,  preactivation = config.stem['preactivation'] , use_maxpool = config.stem['use_maxpool'] , use_avgpool = config.stem['use_avgpool'] , dropout = config.stem['dropout'])
    #elif config.stem['model_name'] == 'mobilenetv2':
    #    stem = feature_extract_network.mobilenetv2(fm_mult = config.stem['fm_mult'] , num_classes = config.stem['num_classes'] , input_size = 128 ,dropout = config.stem['dropout'] )
    model_name = config.stem['model_name']
    kwargs = config.stem
    kwargs.pop('model_name')
    stem = eval( 'feature_extract_network.' + model_name)(**kwargs)

    last_epoch = -1 
    if config.train['resume'] is not None:
        strict = True
        if config.train['pretrained']:
            pre_dim = next(stem.fc2.parameters()).shape[1]
            stem.fc2 = None
            _ = resume_model( stem , config.train['resume'] , epoch = config.train['resume_epoch'] , strict = False ) 
            stem.fc2 = linear( pre_dim , config.stem['num_classes'] , use_batchnorm = False )
            last_epoch = -1
        else:
            last_epoch = resume_model( stem , config.train['resume'] , epoch = config.train['resume_epoch'] , strict = True ) 

    stem =  stem.cuda()

    assert config.train['optimizer'] in ['Adam' , 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam( stem.parameters() , config.train['learning_rate']  ,  weight_decay = config.loss['weight_l2_reg']) 
    if config.train['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD( stem.parameters() ,  config.train['learning_rate'] , weight_decay = config.loss['weight_l2_reg'] , momentum = config.train['momentum'] , nesterov = config.train['nesterov'] )

            

    if config.train['resume_optimizer'] is not None:
        last_epoch = resume_optimizer( optimizer , stem, config.train['resume_optimizer'] , epoch = config.train['resume_epoch'])


    #print( optimizer.param_groups[0]['initial_lr'] )
    if config.train['use_lr_scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer , config.train['lr_scheduler_milestones'] , last_epoch = last_epoch )


    

    set_requires_grad( stem , True )

    cross_entropy = nn.CrossEntropyLoss( ).cuda()

    t = time.time()
    
    
    train_loss_epoch_list = []
    pre_train_loss = 1000

    train_acc_log_list , train_loss_log_list = [] , [] 

    for epoch in range( last_epoch + 1  , config.train['num_epochs'] ):
        best_val_acc = 0 
        best_model = None
        lr_scheduler.step()
        for step , batch in enumerate( train_dataloader ):
            # warm up learning rate
            #if config.train['resume_optimizer'] is None and epoch == last_epoch + 1  :
            #    optimizer.param_groups[0]['lr'] = lr_warmup(step + 1  , config.train['warmup_length'] ) * config.train['learning_rate']

            for k in batch:
                batch[k] = Variable( batch[k].cuda(async =  True) ,requires_grad = False )
            set_requires_grad( stem , True)
            predicts , features = stem( batch['img'] , use_dropout = True  )
            train_acc , train_loss  = compute_loss(predicts  ,  batch['label'] )

            train_acc_log_list.append( train_acc.data.cpu().numpy()[0] )
            train_loss_log_list.append( train_loss.data.cpu().numpy()[0] )
            train_loss_epoch_list.append( train_loss.data.cpu().numpy()[0] )

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            tb.add_scalar( 'loss' , train_loss.data.cpu().numpy() , epoch*len(train_dataloader) + step , 'train') 
            tb.add_scalar( 'acc' , train_acc.data.cpu().numpy() , epoch*len(train_dataloader) + step , 'train' )
            tb.add_scalar( 'lr' , optimizer.param_groups[0]['lr'] , epoch*len(train_dataloader) , 'train')

            if  step % config.train['log_step'] == 0 :
                set_requires_grad( stem ,False)
                tt = time.time()
                if not config.test_time :
                    acc_num_list , loss_list = [] , []
                    it = iter(val_dataloader)
                    for idx , val_batch in enumerate(val_dataloader):
                        for k in val_batch:
                            val_batch[k] = Variable( val_batch[k].cuda(async =  True) ,requires_grad = False )
                        predicts , features = stem( val_batch['img'] , use_dropout = False  )
                        val_acc , val_loss  = compute_loss(predicts  ,  val_batch['label'] )
                        val_acc_num = val_acc * predicts.shape[0]

                        loss_list.append( val_loss )
                        acc_num_list.append( val_acc_num )
                    val_loss = torch.mean( torch.stack( loss_list ))
                    val_acc = torch.sum( torch.stack( acc_num_list )) / len(val_dataloader.dataset)

                    train_loss = np.mean( np.stack( train_loss_log_list ) )
                    train_acc = np.mean( np.stack( train_acc_log_list ))

                    train_loss_log_list , train_acc_log_list = [] , [] 

                    tb.add_scalar( 'loss' , val_loss.data.cpu().numpy() , epoch*len(train_dataloader) + step , 'val') 
                    tb.add_scalar( 'acc' , val_acc.data.cpu().numpy() , epoch*len(train_dataloader) + step , 'val' )

                    #if best_val_acc < val_acc :
                    #    best_val_acc = val_acc
                    #    best_model = copy.copy( stem )

                    log_msg = "epoch {} , step {} / {} , train_loss {:.5f}, train_acc {:.2%} , val_loss {:.5f} , val_acc {:.2%} {:.1f} imgs/s".format(epoch,step,len(train_dataloader) - 1,train_loss,train_acc,val_loss.data.cpu().numpy()[0],val_acc.data.cpu().numpy()[0],config.train['log_step']*config.train['batch_size']/(tt -t)) 
                    print(log_msg )
                    log_file.write(log_msg +'\n')
                    
                    #print( torch.max( predicts , 1  )[1][:5] )
                else:
                    print( "epoch {} , step {} / step {} , data {:.3f}s , mv_to_cuda {:.3f}s forward {:.3f}s acc {:.3f}s loss {:.3f}s , backward {:.3f}s".format(epoch,step,len(train_dataloader) , t1 - t0 , t2 -t1 , t3 - t2 , t4 - t3 , t5 - t4 , t6 - t5) )
                t = tt
        #optimizer.param_groups[0]['lr'] *= config.train['learning_rate_decay'] 
        temp_train_loss = np.mean( np.stack( train_loss_epoch_list  ))

        train_loss_epoch_list = [] 
        train_loss_log_list = []
        train_acc_log_list = []
        #if config.train['auto_adjust_lr']:
        #    auto_adjust_lr( optimizer , pre_train_loss , temp_train_loss )
        #pre_train_loss = temp_train_loss
            
        save_model( stem , tb.path , epoch )
        save_optimizer( optimizer , stem , tb.path , epoch )
        print("Save done in {}".format( tb.path ) )
