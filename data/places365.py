from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import memcache
import io

class dataset_places365(Dataset):
    def __init__(self,data_list_path,rootdir,transform,is_use_memcache=True):
        super(dataset_places365,self).__init__()
        self.data_list=np.genfromtxt(data_list_path,dtype=str,delimiter=' ')
        assert self.data_list.shape[1] == 2
        self.rootdir=rootdir
        self.transform=transform
        self.is_use_memcache=is_use_memcache

        if self.is_use_memcache:
            self.memcache = memcache.Client(['127.0.0.1:11211'])

    def __len__(self):
        return self.data_list.shape[0]

    def mc_get(self,image_path):
        mc = self.memcache
        mc_str =mc.get(image_path)
        if mc_str == None:
            image = Image.open(image_path)
            f = open(image_path, 'rb')
            bin_str = f.read()
            mc.set(image_path, bin_str)
            f.close()
        else:
            img_buf = io.BytesIO(mc_str)
            image = Image.open(img_buf)
        return image

    def __getitem__(self,item):

        rootdir = self.rootdir
        img_path=os.path.join(rootdir , self.data_list[item][0])
        assert os.path.isfile(img_path)
        label=int(self.data_list[item][1])
        if self.is_use_memcache:
            img= self.mc_get(img_path)
        else:
            img= Image.open(img_path)
        img = img.convert('RGB')
        if self.transform != None:
            img=self.transform(img)
        return img,label

if __name__ == '__main__':

    # transform=None
    is_use_memcache=False

    from augmentation import Cutout,GaussianBlur
    from torchvision import transforms
    import PIL
    import cv2

    crop_size=224
    input_size=224
    augmentation_list = [transforms.Resize((256,256),interpolation=2),
                            GaussianBlur(p=0.3,radius=2),
                            transforms.RandomCrop((crop_size,crop_size)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                            transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                            transforms.RandomGrayscale(p=0.1)]

    transform_list = [transforms.Resize((input_size,input_size),interpolation=2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                        ]
    Additional_list=[]
    Additional_list.append(Cutout(n_holes=1, length=32))

    train_transform=transforms.Compose(augmentation_list+transform_list+Additional_list)

    train_dst = dataset_places365('/Users/apple/Documents/scene_classification/data/try_data.lst','/Users/apple/Desktop/',train_transform,is_use_memcache=is_use_memcache)
    while  True:
        img_tensor = train_dst[0][0].data.numpy()
        print(img_tensor.shape)
        img = np.transpose(img_tensor,(1,2,0))
        cv2.imshow('ss',img)
        cv2.waitKey(0)





