import os
from re import L
from sre_constants import GROUPREF_LOC_IGNORE
from tkinter import W
import numpy as np
import cv2
import albumentations
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random
from skimage.metrics import structural_similarity
from tqdm import tqdm

class GOPDATA(Dataset):
    def __init__(self,
                 data_csv, 
                 data_root,
                 gop_min_length=2,
                 gop_max_length=10,
                 ssim_thres=0.8,
                 size=None, 
                 random_crop=False, 
                 interpolation="bicubic",
                 n_labels=182, 
                 coord=False,
                 no_crop=False,
                 no_rescale=False
                 ):

        self.n_labels = n_labels
        self.data_csv = data_csv
        self.data_root = data_root
        self.gop_min_length = gop_min_length
        self.gop_max_length = gop_max_length
        self.ssim_thres = ssim_thres
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._gop()
        self._length = len(self.gop)

        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths]
        }
        self.coord = coord
        if self.coord:
            print("Cylinderical coordinate for 360 image.")

        size = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size, # Sun360 images are 256x512
                                                                 interpolation=self.interpolation)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image", "masked_image": "image", "binary_mask": "image"})
                
            self.preprocessor = self.cropper
        self.no_crop = no_crop
        self.no_rescale = no_rescale

    def __len__(self):
        return self._length
    

    def _gop(self,):

        gop = list()
        sub_gop = list()
        prev = cv2.imread(os.path.join(self.data_root, self.data_csv[0]))
        sub_gop.append(prev)
        for frame in tqdm(self.data_csv[1:]):
            nex = cv2.imread(os.path.join(self.data_root, frame))
            ssim = structural_similarity(prev, nex, channel_axis=-1)

            # if ssim is large, same sub gop, otherwise start a new one
            if ssim >= self.ssim_thres:
                sub_gop.append(nex)
            else:
                if len(sub_gop) >= self.gop_min_length:
                    gop.append(sub_gop)
                sub_gop = list()
                sub_gop.append(nex)
        
        self.gop = gop
    

    def process_gop(self, frame_list):

        frame_list = [os.path.join(self.data_root, frame_path) for frame_path in frame_list]

        images = list()
        

        gop_data = dict()
        for frame_path in frame_list:
            image = Image.open(frame_path)
            image = image if image.mode == 'RGB' else image.convert("RGB")
            if not self.no_rescale and self.size is not None: # Default True. False when refine net training.
                image = self.image_rescaler(image=image)["image"]
            
            masked_image, binary_image = self.masking(image)

            # add coord 
            h, w, _ = image.shape
            coord = np.tile(np.arange(h).reshape(h,1,1), (1,w,1)) / (h-1) * 2 - 1 # -1~ 1
            # sin, cos
            coord = self.add_cylinderical(coord) # -1 ~ 1
            gop_data['frame_path'] = (image, coord, masked_image, binary_mask)

        # init split point
        split_point = torch.randint(0, w,(1,)) # w 
        ret_batch = defaultdict(list)
        for frame_path, (image, coord, masked_image, binary_mask) in gop_data.items():
            # rotation augmentation 
            image, coord, masked_image, binary_mask = self.rotation_augmentation(image, coord, masked_image, binary_mask, split_point)

            if not self.no_crop and self.size is not None: # self.no_crop = True when training Transformer with 1:2 images, or icip(256x512)
                processed = self.cropper(image=image, coord=coord, masked_image=masked_image, binary_mask=binary_mask)
                image = processed['image']
                coord = processed['coord']
                masked_image = processed['masked_image']
                binary_mask = processed['binary_mask']
            
            image = (image/127.5 - 1.0).astype(np.float32)
            masked_image = (masked_image/127.5 - 1.0).astype(np.float32)

            ret_batch['path'].append(frame_path)
            ret_batch['image'].append(image)
            ret_batch['masked_image'].append(masked_image)
            ret_batch['binary_mask'].append(binary_mask)
            ret_batch['coord'].append(coord)
            ret_batch['concat_input'].append(np.concatenate((masked_image, coord, binary_mask), axis=2))
        
        return ret_batch

    def __getitem__(self, i):
        frame_list = self.gop[i]
        return self.process_gop(frame_list)


    def rotation_augmentation(self, im, coord, masked_im, binary_mask, split_point=None):
        if split_point is None:
            split_point = torch.randint(0, im.shape[1],(1,)) # w 
        #split_point = im.shape[1] // 2
        im = np.concatenate( (im[:,split_point:,:], im[:,:split_point,:]), axis=1 )
        coord = np.concatenate( (coord[:,split_point:,:], coord[:,:split_point,:]), axis=1 )
        masked_im = np.concatenate( (masked_im[:,split_point:,:], masked_im[:,:split_point,:]), axis=1 )
        binary_mask = np.concatenate( (binary_mask[:,split_point:,:], binary_mask[:,:split_point,:]), axis=1 )
        return im, coord, masked_im, binary_mask

    def masking(self, im):
        h, w, c = im.shape
        binary_mask = np.zeros((h,w,1))
        # random mask position
        # margin_h = int( (180 - torch.randint(70, 95, (1,)) ) / 360 * h )
        margin_h = int( (180 - 80 ) / 360 * h )
        #print(margin_h, (h - margin_h))
        binary_mask[margin_h:(h - margin_h), int(w/4):int(w/4)*3, :] = 1.
        canvas = np.ones_like(im) * 127.5
        canvas = im * binary_mask + canvas * (1 - binary_mask)
        return canvas, binary_mask

    def add_cylinderical(self, coord):
        h, w, _ = coord.shape
        sin_img = np.sin(np.radians(np.arange(w) / w * 360))
        sin_img[np.abs(sin_img) < 1e-6] = 0
        sin_img = np.tile(sin_img, (h,1))[:,:,np.newaxis]
        cos_img = np.cos(np.radians(np.arange(w) / w * 360))
        cos_img[np.abs(cos_img) < 1e-6] = 0
        cos_img = np.tile(cos_img, (h,1))[:,:,np.newaxis]
        return np.concatenate((coord, sin_img, cos_img), axis=2)
        
