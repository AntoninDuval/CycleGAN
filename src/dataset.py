import torch
from os import listdir
from os.path import isfile, join
import PIL

WORKING_SHAPE = (256, 256)

class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        """
          Init the dataset
          path: OS path where the dataset is located
          transform : transformation to apply to the dataset

        """
        super(DomainDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.items = [f for f in listdir(path) if isfile(join(path, f))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_raw = PIL.Image.open(f'{self.path}/{self.items[idx]}')
        # for conversion to 0-1 range
        img_raw = img_raw.convert('L') if img_raw.mode != 'L' else img_raw
        return self.transform(img_raw.resize(WORKING_SHAPE))