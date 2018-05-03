from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import face_alignment
from math import floor
from utils import landmarks_68_to_5

def process(img  , landmarks_5pts):
    batch = {}
    name = ['left_eye','right_eye','nose','mouth']
    patch_size = {
            'left_eye':(40,40),
            'right_eye':(40,40),
            'nose':(40,32),
            'mouth':(48,32),
    }
    landmarks_5pts[3,0] =  (landmarks_5pts[3,0] + landmarks_5pts[4,0]) / 2.0
    landmarks_5pts[3,1] = (landmarks_5pts[3,1] + landmarks_5pts[4,1]) / 2.0

    # crops
    for i in range(4):
        x = floor(landmarks_5pts[i,0])
        y = floor(landmarks_5pts[i,1])
        batch[ name[i] ] = img.crop( (x - patch_size[ name[i] ][0]//2 + 1 , y - patch_size[ name[i] ][1]//2 + 1 , x + patch_size[ name[i] ][0]//2 + 1 , y + patch_size[ name[i] ][1]//2 + 1 ) )



    return batch

class TrainDataset( Dataset):
    def __init__( self , img_list ):
        super(type(self),self).__init__()
        self.img_list = img_list
    def __len__( self ):
        return len(self.img_list)
    def __getitem__( self , idx ):
        #filename processing
        batch = {}
        img_name = self.img_list[idx].split('/')
        img_frontal_name = self.img_list[idx].split('_')
        img_frontal_name[-2] = '051'
        img_frontal_name = '_'.join( img_frontal_name ).split('/')
        batch['img'] = Image.open( '/'.join( img_name ) )
        batch['img32'] = Image.open( '/'.join( img_name[:-2] + ['32x32' , img_name[-1] ] ) )
        batch['img64'] = Image.open( '/'.join( img_name[:-2] + ['64x64' , img_name[-1] ] ) )
        batch['img_frontal'] = Image.open( '/'.join(img_frontal_name) )
        batch['img32_frontal'] = Image.open( '/'.join( img_frontal_name[:-2] + ['32x32' , img_frontal_name[-1] ] ) )
        batch['img64_frontal'] = Image.open( '/'.join( img_frontal_name[:-2] + ['64x64' , img_frontal_name[-1] ] ) )
        patch_name_list = ['left_eye','right_eye','nose','mouth']
        for patch_name in patch_name_list:
            batch[patch_name] = Image.open( '/'.join(img_name[:-2] + ['patch' , patch_name , img_name[-1] ]) ) 
            batch[patch_name+'_frontal'] = Image.open( '/'.join(img_frontal_name[:-2] + ['patch' , patch_name , img_frontal_name[-1] ]) )
        totensor = transforms.ToTensor()

        for k in batch:
            batch[k] = totensor( batch[k] ) 
            batch[k] = batch[k] *2.0 -1.0

        batch['label'] = int( self.img_list[idx].split('/')[-1].split('_')[0] )
        return batch

        

class PretrainDataset( Dataset):
    def __init__( self , img_list ):
        super(type(self),self).__init__()
        self.img_list = img_list
    def __len__( self):
        return len(self.img_list)
    def __getitem__(self,idx):
        batch = {}

        totensor = transforms.ToTensor()
        img = Image.open( self.img_list[idx] )
        img = totensor( img )
        img = img*2.0 - 1.0

        batch['img'] = img
        batch['label'] = int( self.img_list[idx].split('/')[-1].split('_')[0] )
        return  batch

class TestDataset( Dataset):
    def __init__( self , img_list , lm_list):
        super(type(self),self).__init__()
        self.img_list = img_list
        self.lm_list = lm_list
        assert len(img_list) == len(lm_list)
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self,idx):
        img_name = self.img_list[idx]
        img = Image.open( img_name )

        lm = np.array( self.lm_list[idx].split(' ') , np.float32 ).reshape(-1,2)
        lm = landmarks_68_to_5( lm )
        for i in range(5):
            lm[i][0] *= 128/img.width 
            lm[i][1] *= 128/img.height
        img = img.resize( (128,128) , Image.LANCZOS)
        batch = process( img , lm )
        batch['img'] = img
        batch['img64'] = img.resize( (64,64) , Image.LANCZOS )
        batch['img32'] = batch['img64'].resize( (32,32) , Image.LANCZOS )
        to_tensor = transforms.ToTensor() 
        for k in batch:
            batch[k] = to_tensor( batch[k] )
            batch[k] = batch[k] * 2.0 - 1.0
        return batch


