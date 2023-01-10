import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import pathlib

class Div2kDataset(Dataset):
    def __init__(self,
                 path,
                 mode,
                 cache_path=None,
                 patch_size=128,
                 dataset_repeats=1,
                 interpolation='bilin',
                 augmentation=None,
                 test_percent=0.10,
                 upsample_size=4,
                 ):
        super().__init__()
        assert interpolation in ['bilin', 'bicub', None]
        assert mode in ['train', 'val']
        assert augmentation in [None]
        self.patch_size = patch_size
        self.interpolation = interpolation
        self.mode = mode
        self.augmentation = augmentation
        self.dataset_repeats = dataset_repeats

        self.cache_path = cache_path
        self.image_path = path

        examples = sorted(list(os.listdir(path)))
        num_val = int(len(examples) * test_percent)
        num_train = len(examples) - num_val

        self.train = examples[:num_train]
        self.val = examples[num_train: num_train + num_val]
        self.upsample_size = upsample_size
        # if not os.path.exists(self.cache_path):
        pathlib.Path(self.cache_path).mkdir(parents=True, exist_ok=True)

    def __len__(self):
        l = len(self.train) if self.mode == 'train' else len(self.val)
        return l * self.dataset_repeats
        # return len(self.landmarks_frame)

    def __getitem__(self, idx):
        nsamps = len(self.train if self.mode == 'train' else self.val)
        if idx >= self.dataset_repeats * nsamps:
            raise StopIteration

        if self.mode == 'val':
            random.seed(idx)

        ex_name = self.train[idx % nsamps] if self.mode == 'train' else self.val[idx % nsamps]

        if os.path.isfile(os.path.join(self.cache_path, f"{ex_name}.pt")):
            image = torch.load(os.path.join(self.cache_path, f"{ex_name}.pt"), map_location=torch.device('cpu'))
        else:
            image = Image.open(os.path.join(self.image_path, ex_name))
            image = torchvision.transforms.functional.to_tensor(image)
            if self.cache_path is not None:
                torch.save(image, os.path.join(self.cache_path, f"{ex_name}.pt"))

        x = random.randint(0, image.shape[-2] - self.patch_size)
        y = random.randint(0, image.shape[-1] - self.patch_size)
        image =  image[:, x:x+self.patch_size, y:self.patch_size + y]
        lr = torch.nn.functional.interpolate(image[None], size=self.patch_size//self.upsample_size, mode='bilinear', antialias='bilinear')[0]
        return image, lr

#
class Div2kDataModule(pl.LightningDataModule):
    def __init__(self,
                 path,
                 cache_path=None,
                 patch_size=128,
                 dataset_repeats=1,
                 interpolation='bilin',
                 augmentation=None,
                 test_percent=0.10,
                 upsample_size=4,
                 batch_size=8,
                 num_workers=8
                 ):
        super().__init__()
        assert interpolation in ['bilin', 'bicub', None]
        self.patch_size = patch_size
        self.interpolation = interpolation
        self.augmentation = augmentation
        self.dataset_repeats = dataset_repeats
        self.cache_path = cache_path
        self.image_path = path
        self.test_precent = test_percent
        self.upsample_size = upsample_size
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage: str):

        self.div2k = Div2kDataset(self.image_path,
                                  'train',
                                  cache_path=self.cache_path,
                                  patch_size=self.patch_size,
                                  dataset_repeats=self.dataset_repeats,
                                  interpolation=self.interpolation,
                                  augmentation=self.augmentation,
                                  test_percent=self.test_precent,
                                  upsample_size=self.upsample_size)
        self.div2k_test = Div2kDataset(self.image_path,
                                      'val',
                                      cache_path=self.cache_path,
                                      patch_size=self.patch_size,
                                      dataset_repeats=1,
                                      interpolation=self.interpolation,
                                      augmentation=self.augmentation,
                                      test_percent=self.test_precent,
                                      upsample_size=self.upsample_size)


    def train_dataloader(self):
        return DataLoader(self.div2k, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=4)

    def val_dataloader(self):
        return DataLoader(self.div2k_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.div2k_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.div2k_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass