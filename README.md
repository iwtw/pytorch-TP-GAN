# TP-GAN
pytorch replicate of TP-GAN ["Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis"](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Beyond_Face_Rotation_ICCV_2017_paper.pdf)
## what's different from the official code

- I use wasserstein-GP as adversial loss
- I tried adopting modified ReNet18 or MobilNetV2 to extract features to compute perceptual loss (idendity preserving loss in the original paper)
- remove batch normalization layers
- remove the last tanh activation in generator
- change the first conv and the first residual block in decoder of generator's kernel size from 2 to 3

## usage

to train feature extract models
```
vim pretrain_config.py #set options
python pretrain.py
```
to train TP-GAN 
```
vim config.py #set options
python train.py
```
to test TP-GAN
```
python test.py $args
```

##some other implementations

 - [official tensorflow implementation](https://github.com/HRLTY/TP-GAN)
 - [tensorflow replicate](https://github.com/ddddwee1/sul/tree/master/sample/tpgan)
