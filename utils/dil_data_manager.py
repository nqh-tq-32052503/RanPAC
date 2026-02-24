import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR224, iImageNetR, iImageNetA, CUB, omnibenchmark, vtab, cars, core50, cddb, domainnet
from dildatasets import get_dataset, get_all_datasets
from argparse import Namespace

class DILDataManager(object):
    def __init__(self, args, dataset_name, shuffle, seed, init_cls, increment,use_input_norm=False):
        self.dataset_name = dataset_name
        self.args = Namespace(**args) 
        dataset = get_dataset(self.args, dataset_name)
        self.dataset = dataset
        train_loader, test_loader = self.dataset.get_all_data_loaders()
        self.train_dataset = train_loader.dataset
        self.test_dataset = test_loader.dataset
        

    @property
    def nb_tasks(self):
        return 1

    def get_task_size(self, task):
        return self.dataset.N_CLASSES

    def get_total_classnum(self):
        return self.dataset.N_CLASSES

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        if source == "train":
            return DummyDataset(self.train_dataset)
        elif source == "test":
            return DummyDataset(self.test_dataset)
        else:
            raise ValueError("Unknown data source {}.".format(source))

class DummyDataset(Dataset):
    def __init__(self, dataset):
        self.core_dataset = dataset
        self.default_transform = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.core_dataset)

    def __getitem__(self, idx):
        data = self.core_dataset[idx]
        if len(data) == 2:
            img, target = data
        elif len(data) == 3:
            img, target, not_aug_img = data
        elif len(data) == 4:
            img, target, not_aug_img, _ = data 
        else:
            print("[ERROR], len(data)=", len(data))
        if img.shape[2] != 224:
            img = self.default_transform(img)
        if img.shape[0] != 3:
            img = img.expand(3, -1, -1)
        return idx, img, target