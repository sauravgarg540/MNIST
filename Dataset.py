import torch
import idx2numpy
import numpy as np
from torch.utils.data import Dataset


class ToTorchFormatTensor(object):

    def __call__(self, image):
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).contiguous()
        return image.float()


class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, phase ='train'):
        self.images = idx2numpy.convert_from_file(img_dir).copy()
        self.labels = idx2numpy.convert_from_file(label_dir).copy()
        self.transform = transform

        if phase == 'train':
            self.images = idx2numpy.convert_from_file(img_dir)[:50000].copy()
            self.labels = idx2numpy.convert_from_file(label_dir)[:50000].copy()
        elif phase == 'validation':
            self.images = idx2numpy.convert_from_file(img_dir)[50000:].copy()
            self.labels = idx2numpy.convert_from_file(label_dir)[50000:].copy()
        else:
            self.images = idx2numpy.convert_from_file(img_dir).copy()
            self.labels = idx2numpy.convert_from_file(label_dir).copy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image= self.transform(image)
        label = torch.tensor(float(label)).long()
        return image, label