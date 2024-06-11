import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HarvardSkinCancerDatasset(Dataset):

    def __init__(self, df: pd.DataFrame, transforms=None, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index])
        y = torch.tensor(self.df['lesion_type_id'][index])

        if self.transforms:
            X = self.transforms(X)

        return X, y
