'''FEC Dataset class'''

import pprint
import pandas as pd
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class FecData(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv("./Data/triplet/pd_triplet_data.csv")
        self.data = self.pd_data.to_dict("list")
        self.data_anc = self.data['anchor']
        self.data_pos = self.data["postive"]
        self.data_neg = self.data["negative"]

    def __len__(self):
        return len(self.data["anchor"])

    def __getitem__(self, index):
        anc_list = self.data_anc[index]
        anc_img = Image.open(anc_list)
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')

        pos_list = (self.data_pos[index])
        pos_img = Image.open(pos_list)
        if pos_img.getbands()[0] != 'R':
            pos_img = pos_img.convert('RGB')

        neg_list = (self.data_neg[index])
        neg_img = Image.open(neg_list)
        if neg_img.getbands()[0] != 'R':
            neg_img = neg_img.convert('RGB')


        if self.transform is not None:
            anc_img = self.transform(anc_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        return anc_img, pos_img, neg_img


class FecTestData(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, transform=None):
        self.transform = transform
        self.pd_test_data = pd.read_csv("./Data/triplet/pd_triplet_data_test.csv")
        self.data = self.pd_test_data.to_dict("list")
        self.data_anc = self.data['anchor']
        self.data_pos = self.data["postive"]
        self.data_neg = self.data["negative"]

    def __len__(self):
        return len(self.data["anchor"])

    def __getitem__(self, index):
        anc_list = self.data_anc[index]
        anc_img = Image.open(anc_list)
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')
        
        pos_list = self.data_pos[index]
        pos_img = Image.open(pos_list)
        if pos_img.getbands()[0] != 'R':
            pos_img = pos_img.convert('RGB')

        neg_list = self.data_neg[index]
        neg_img = Image.open(neg_list)
        if neg_img.getbands()[0] != 'R':
            neg_img = neg_img.convert('RGB')


        if self.transform is not None:
            anc_img = self.transform(anc_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        return anc_img, pos_img, neg_img
