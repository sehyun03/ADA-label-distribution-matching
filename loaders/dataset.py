import numpy as np
import os
import os.path
from PIL import Image
import torch

def get_scenario(dataset):
    if dataset == "office_home" or dataset == "office_home_new":
        scenario_list = ["A2C","A2P","A2R","C2A","C2P","C2R","P2A","P2C","P2R","R2A","R2C","R2P"]
    elif dataset == "office_home_rsut":
        scenario_list = ["C2P","C2R","P2C","P2R","R2C","R2P"]        
    elif dataset == "domainnet":
        scenario_list = ["R2C","C2S","S2P","C2Q"]
    elif dataset == "visda17":
        scenario_list = ["R2S"]
    else:
        raise NotImplementedError

    return scenario_list

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list

def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list

class Imagelists_VISDA(object):
    def __init__(self, image_list, root="./data/multi/", transform=None, target_transform=None, isalbu=True):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.isalbu = isalbu

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        assert(self.transform is not None)
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.isalbu:
            img = self.transform(image=np.array(img))['image']
        else:
            img= self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        index = torch.tensor(index) if isinstance(index, int) else index
        return img, target, index

    def __len__(self):
        return len(self.imgs)

class PairedAugDataset(Imagelists_VISDA):
    def __init__(self, image_list, root='./data/multi/',
                 transform=None, transform2=None, target_transform=None, test=False):
        super().__init__(image_list, root, transform, target_transform, test)
        assert self.transform is not None, "no transform1"
        assert transform2 is not None, "no transform2"
        assert not test, "paired dataset for testing?"
        self.transform1 = transform
        self.transform2 = transform2
    
    def __getitem__(self, index):
        path = os.path.join(self.root, self.imgs[index])
        img = self.loader(path)
        img = np.array(img)
        img1 = self.transform1(image=img.copy())['image']
        img2 = self.transform2(image=img)['image']

        target = self.labels[index]
        target = target if self.target_transform is None else self.target_transform(target)

        return img1, img2, target, torch.tensor(index)

    def __len__(self):
        return len(self.imgs)