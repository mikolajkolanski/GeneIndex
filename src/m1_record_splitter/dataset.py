from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors

IMG_W = 1920
IMG_H = 1600

image_pre_trans = v2.Compose([
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Resize([IMG_H,IMG_W]),
])

class ScanDatasetPreload(Dataset):
    def __init__(self,image_paths,labels): 
        self.labels = []

        self.images = []

        for image_path,labs in tqdm(list(zip(image_paths,labels)), desc='Loading images to ram'):
            img = Image.open(image_path)
            img.load() # Do not lazy-load
            iw,ih = img.size

            scale_x = IMG_W/iw
            scale_y = IMG_H/ih
            labs = [[x,y,x+w,y+h,act] for [x,y,w,h,act] in labs]
            labs = [[x1*scale_x,y1*scale_y,x2*scale_x,y2*scale_y,act] for [x1,y1,x2,y2,act] in labs]

            img_t = image_pre_trans(img)
            self.images.append(img_t)
            self.labels.append(labs)
        

    def __getitem__(self,idx):
        return tv_tensors.Image(self.images[idx]), {
            "boxes": tv_tensors.BoundingBoxes([i[0:4] for i in self.labels[idx]], format="XYXY", canvas_size=(IMG_H,IMG_W)),
            "labels": torch.tensor([1 if i[-1] is None else 2 for i in self.labels[idx]])
        }
    def __len__(self):
        return len(self.images)

class ScanDataset(Dataset):
    def __init__(self,image_paths,labels): 
        self.labels = labels
        self.image_paths = image_paths
        self.len = len(image_paths)
        

    def __getitem__(self,idx):
        img = Image.open(self.image_paths[idx])
        img.load()
        iw,ih = img.size

        scale_x = IMG_W/iw
        scale_y = IMG_H/ih
        labs = [[x,y,x+w,y+h,act] for [x,y,w,h,act] in self.labels[idx]]
        labs = [[x1*scale_x,y1*scale_y,x2*scale_x,y2*scale_y,act] for [x1,y1,x2,y2,act] in labs]

        img_t = image_pre_trans(img)

        return tv_tensors.Image(img_t), {
            "boxes": tv_tensors.BoundingBoxes([i[0:4] for i in labs], format="XYXY", canvas_size=(IMG_H,IMG_W)),
            "labels": torch.tensor([1 if i[-1] is None else 2 for i in labs])
        }
    def __len__(self):
        return self.len