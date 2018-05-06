train =  {}
train['img_list'] = './img.list'
train['learning_rate'] = 1e-4
train['num_epochs'] = 200
train['batch_size'] = 20
train['log_step'] = 1000
train['resume_model'] = None
train['resume_optimizer'] = None

G = {}
G['zdim'] = 64
G['use_residual_block'] = False
G['use_batchnorm'] = False
G['num_classes'] = 347

D = {}
D['use_batchnorm'] = False

loss = {}
loss['weight_gradient_penalty'] = 10
loss['weight_128'] = 1.0
loss['weight_64'] = 1.0
loss['weight_32'] = 1.5
loss['weight_pixelwise'] = 1.0
loss['weight_pixelwise_local'] = 3.0

loss['weight_symmetry'] = 3e-1
loss['weight_adv_G'] = 1e-3
loss['weight_identity_preserving'] = 3e1
loss['weight_total_varation'] = 1e-3
loss['weight_cross_entropy'] = 1e1

feature_extract_model = {}
feature_extract_model['resume'] =  'save/feature_extract_model/resnet18/try_1'
