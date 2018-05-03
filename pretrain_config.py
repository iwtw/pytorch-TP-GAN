#train_config
test_time = False

num_classes = 347

train = {}
train['train_img_list'] = 'pretrain_train.list'
train['val_img_list'] = 'pretrain_val.list'

train['batch_size'] = 128
train['num_epochs'] = 35
train['log_step'] = 100

train['optimizer'] = 'SGD'
train['learning_rate'] = 1e-1
train['momentum'] = 0.9
train['nesterov'] = True 
train['warmup_length'] = 0
train['learning_rate_decay'] = 1.0
train['auto_adjust_lr'] = False

train['sub_dir'] = 'feature_extract_model'
train['pretrained'] = None
train['resume'] = None
train['resume_optimizer'] = None
train['resume_epoch'] = None  #None means the last epoch

train['use_lr_scheduler'] = True
train['lr_scheduler_milestones'] = [10,20,30]



stem = {}
model_name = ['mobilenetv2' , 'resnet18' ]
stem['model_name'] = 'mobilenetv2'
stem['num_classes'] = num_classes

assert stem['model_name'] in model_name
if stem['model_name'] == 'mobilenetv2':
    stem['fm_mult'] = 1.0
    stem['dropout'] = 0.5
    stem['input_size'] = 128
elif stem['model_name'] == 'resnet18':
    stem['fm_mult'] = 0.5
    stem['feature_layer_dim'] = None
    stem['preactivation'] = True
    stem['use_batchnorm'] = True
    stem['use_avgpool'] = True
    stem['use_maxpool'] = False
    stem['dropout'] = 0.5





loss = {}
loss['weight_l2_reg'] = 1e-4
