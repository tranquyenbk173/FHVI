import os
from functools import partial
from typing import Optional, Sequence
import torch.utils.data as data

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    DTD,
    STL10,
    FGVCAircraft,
    Flowers102,
    Food101,
    ImageFolder,
    OxfordIIITPet,
    StanfordCars,
    Caltech101,
    SVHN,
    SUN397,
    EuroSAT,    
)

from PIL import Image

DATASET_DICT = {
    "cifar10": [
        partial(CIFAR10, train=True, download=True),
        partial(CIFAR10, train=False, download=True),
        partial(CIFAR10, train=False, download=True),
        10,
    ],
    "cifar100": [
        partial(CIFAR100, train=True, download=True),
        partial(CIFAR100, train=False, download=True),
        partial(CIFAR100, train=False, download=True),
        100,
    ],
    "caltech101": [
        partial(Caltech101, train=True, download=True),
        partial(Caltech101, train=False, download=True),
        partial(Caltech101, train=False, download=True),
        102,
    ],
    "flowers102": [
        partial(Flowers102, split="train", download=True),
        partial(Flowers102, split="val", download=True),
        partial(Flowers102, split="test", download=True),
        102,
    ],
    "food101": [
        partial(Food101, split="train", download=True),
        partial(Food101, split="test", download=True),
        partial(Food101, split="test", download=True),
        101,
    ],
    "pets37": [
        partial(OxfordIIITPet, split="trainval", download=True),
        partial(OxfordIIITPet, split="test", download=True),
        partial(OxfordIIITPet, split="test", download=True),
        37,
    ],
    "stl10": [
        partial(STL10, split="train", download=True),
        partial(STL10, split="test", download=True),
        partial(STL10, split="test", download=True),
        10,
    ],
    "dtd": [
        partial(DTD, split="train", download=True),
        partial(DTD, split="val", download=True),
        partial(DTD, split="test", download=True),
        47,
    ],
    "aircraft": [
        partial(FGVCAircraft, split="train", download=True),
        partial(FGVCAircraft, split="val", download=True),
        partial(FGVCAircraft, split="test", download=True),
        100,
    ],
    "cars": [
        partial(StanfordCars, split="train", download=True),
        partial(StanfordCars, split="test", download=True),
        partial(StanfordCars, split="test", download=True),
        196,
    ],
    "svhn":  [
        partial(SVHN, split="train", download=True),
        partial(SVHN, split="test", download=True),
        partial(SVHN, split="test", download=True),
        10,
    ],
    "SUN397": [
        partial(SUN397, split="train", download=True),
        partial(SUN397, split="test", download=True),
        partial(SUN397, split="test", download=True),
        397,
    ],

    "patch_camelyon": [
        partial(SUN397, split="train", download=True),
        partial(SUN397, split="test", download=True),
        partial(SUN397, split="test", download=True),
        2
    ],
    "eurosat": [
        partial(EuroSAT, split="train", download=True),
        partial(EuroSAT, split="test", download=True),
        partial(EuroSAT, split="test", download=True),
        10
    ],
    "resisc45": [
        partial(SUN397, split="train", download=True),
        partial(SUN397, split="test", download=True),
        partial(SUN397, split="test", download=True),
        45,
    ],
    "retinopathy": [
        partial(SUN397, split="train", download=True),
        partial(SUN397, split="test", download=True),
        partial(SUN397, split="test", download=True),
        5,
    ],
    "clevrcount": [
        partial(SUN397, split="train", download=True),
        partial(SUN397, split="test", download=True),
        partial(SUN397, split="test", download=True),
        8,
    ],
    "clevrdist": [
        partial(SUN397, split="train", download=True),
        partial(SUN397, split="test", download=True),
        partial(SUN397, split="test", download=True),
        6,
    ],
    "dmlab": [
        partial(SUN397, split="train", download=True),
        partial(SUN397, split="test", download=True),
        partial(SUN397, split="test", download=True),
        6,
    ],
    "kitti": [
        partial(SUN397, split="train", download=True),
        partial(SUN397, split="test", download=True),
        partial(SUN397, split="test", download=True),
        4,
    ],
    "dsprites_loc": [
        partial(SUN397, split="train", download=True),
        partial(SUN397, split="test", download=True),
        partial(SUN397, split="test", download=True),
        16,
    ],
    "dsprites_ori": [
        partial(SUN397, split="train", download=True),
        partial(SUN397, split="test", download=True),
        partial(SUN397, split="test", download=True),
        16,
    ],

    "smallnorb_azi": [
        partial(SUN397, split="train", download=True),
        partial(SUN397, split="test", download=True),
        partial(SUN397, split="test", download=True),
        18,
    ],

    "smallnorb_ele": [
        partial(SUN397, split="train", download=True),
        partial(SUN397, split="test", download=True),
        partial(SUN397, split="test", download=True),
        9,
    ],
    
}

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str = "cifar10",
        root: str = "data/",
        num_classes: Optional[int] = None,
        size: int = 224,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        flip_prob: float = 0.5,
        rand_aug_n: int = 0,
        rand_aug_m: int = 9,
        erase_prob: float = 0.0,
        use_trivial_aug: bool = False,
        mean: Sequence = (0.5, 0.5, 0.5),
        std: Sequence = (0.5, 0.5, 0.5),
        batch_size: int = 32,
        workers: int = 4,
    ):
        """Classification Datamodule

        Args:
            dataset: Name of dataset. One of [custom, cifar10, cifar100, flowers102
                     food101, pets37, stl10, dtd, aircraft, cars]
            root: Download path for built-in datasets or path to dataset directory for custom datasets
            num_classes: Number of classes when using a custom dataset
            size: Crop size
            min_scale: Min crop scale
            max_scale: Max crop scale
            flip_prob: Probability of applying horizontal flip
            rand_aug_n: RandAugment number of augmentations
            rand_aug_m: RandAugment magnitude of augmentations
            erase_prob: Probability of applying random erasing
            use_trivial_aug: Use TrivialAugment instead of RandAugment
            mean: Normalization means
            std: Normalization standard deviations
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.root = root
        self.size = size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.flip_prob = flip_prob
        self.rand_aug_n = rand_aug_n
        self.rand_aug_m = rand_aug_m
        self.erase_prob = erase_prob
        self.use_trivial_aug = use_trivial_aug
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.workers = workers

        # Define dataset
        if self.dataset == "custom":
            assert (
                num_classes is not None
            ), "Must set --data.num_classes when using a custom dataset"
            self.num_classes = num_classes

            self.train_dataset_fn = partial(
                ImageFolder, root=os.path.join(self.root, "train")
            )
            self.val_dataset_fn = partial(
                ImageFolder, root=os.path.join(self.root, "val")
            )
            self.test_dataset_fn = partial(
                ImageFolder, root=os.path.join(self.root, "test")
            )
            print(f"Using custom dataset from {self.root}")
        else:
            pass

            try:
                (
                    self.train_dataset_fn,
                    self.val_dataset_fn,
                    self.test_dataset_fn,
                    self.num_classes,
                ) = DATASET_DICT[self.dataset]
                print(f"Using the {self.dataset} dataset")
                # print(self.train_dataset_fn)
            except:
                raise ValueError(
                    f"{dataset} is not an available dataset. Should be one of {[k for k in DATASET_DICT.keys()]}"
                )

        self.transforms_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (self.size, self.size),
                    scale=(self.min_scale, self.max_scale),
                ),
                transforms.RandomHorizontalFlip(self.flip_prob),
                transforms.TrivialAugmentWide()
                if self.use_trivial_aug
                else transforms.RandAugment(self.rand_aug_n, self.rand_aug_m),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
                transforms.RandomErasing(p=self.erase_prob),
            ]
        )
        self.transforms_test = transforms.Compose(
            [
                transforms.Resize(
                    (self.size, self.size),
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def prepare_data(self):
        if self.dataset != "custom":
            pass
            # self.train_dataset_fn(self.root)
            # self.val_dataset_fn(self.root)
            # self.test_dataset_fn(self.root)
            
    # def train_dataset_fn(root=self.root, transform=self.transforms_train):
    #     return ImageFilelist(root=self.root, flist=root + "/train800.txt",
    #             transform=self.transforms_train)

    def setup(self, stage="fit"):
        if self.dataset == "custom":
            if stage == "fit":
                self.train_dataset = self.train_dataset_fn(
                    transform=self.transforms_train
                )
                self.val_dataset = self.val_dataset_fn(transform=self.transforms_test)
            elif stage == "validate":
                self.val_dataset = self.val_dataset_fn(transform=self.transforms_test)
            elif stage == "test":
                self.test_dataset = self.test_dataset_fn(transform=self.transforms_test)
        else:
            if stage == "fit":
                print('>>>> Stage fit \n \n \n')
                # self.train_dataset = self.train_dataset_fn(
                #     self.root, transform=self.transforms_train, download=False
                # )
                # self.val_dataset = self.val_dataset_fn(
                #     self.root, transform=self.transforms_test, download=False
                # )
                self.train_dataset = ImageFilelist(root=self.root, flist=self.root + "/train800.txt",
                transform=self.transforms_train)
                self.val_dataset = ImageFilelist(root=self.root, flist=self.root + "/val200.txt",
                transform=self.transforms_test)
                self.test_dataset = ImageFilelist(root=self.root, flist=self.root + "/test.txt",
                transform=self.transforms_test)
            elif stage == "validate":
                print('>>>> Stage val \n \n \n')
                # self.val_dataset = self.val_dataset_fn(
                #     self.root, transform=self.transforms_test, download=False
                # )
                self.val_dataset = ImageFilelist(root=self.root, flist=self.root + "/val200.txt",
                transform=self.transforms_test)
            elif stage == "test":
                print('>>>> Stage test \n \n \n')
                # self.test_dataset = self.test_dataset_fn(
                #     self.root, transform=self.transforms_test, download=False
                # )
                self.test_dataset = ImageFilelist(root=self.root, flist=self.root + "/test.txt",
                transform=self.transforms_test)

            

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )
