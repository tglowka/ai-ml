import os
import random
from os.path import join

import numpy
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, imgs_dir):
        self.imgs_dir = imgs_dir
        self.img_paths = []
        self.labels = []
        self.__load_data()
        zipped = list(zip(self.img_paths, self.labels))
        random.shuffle(zipped)
        self.img_paths, self.labels = zip(*zipped)

    def __load_data(self):
        print("Dataset: __load_data")
        for filename in os.listdir(self.imgs_dir):
            self.img_paths.append(join(self.imgs_dir, filename))
            label = int(filename.split("_")[0])
            label = numpy.clip(label, 1, 80)
            self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), antialias=True)])
        img = torch.FloatTensor(transform(img))
        label = torch.Tensor([self.labels[idx]])
        return img, label
